import numpy as np
import pandas as pd
import anndata as ad
from joblib import Parallel, delayed

from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import mode
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Use tqdm.auto for automatic Jupyter/terminal detection
try:
    from tqdm.auto import tqdm, trange
    # Additional import for explicit notebook detection
    from tqdm.notebook import tqdm as tqdm_notebook
    NOTEBOOK_AVAILABLE = True
except ImportError:
    from tqdm import tqdm, trange
    NOTEBOOK_AVAILABLE = False

# Check if we're in a Jupyter environment
def _is_notebook():
    """Check if we're running in a Jupyter notebook"""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
                return True  # Jupyter notebook
        return False
    except ImportError:
        return False

# Choose the appropriate tqdm
if _is_notebook() and NOTEBOOK_AVAILABLE:
    from tqdm.notebook import tqdm as progress_bar
    print("🪐 Jupyter notebook detected - using notebook progress bars")
else:
    from tqdm import tqdm as progress_bar
    print("🖥️  Terminal detected - using standard progress bars")

warnings.filterwarnings('ignore')

# GPU acceleration imports - using PyTorch instead of CuPy
try:
    import torch
    
    # Check for GPU availability (CUDA or MPS)
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        GPU_DEVICE = 'cuda'
        print(f"🚀 CUDA GPU acceleration available (CUDA {torch.version.cuda})")
        print(f"📱 GPU device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        GPU_AVAILABLE = True
        GPU_DEVICE = 'mps'
        print(f"🍎 Apple MPS GPU acceleration available")
        print(f"📱 Metal GPU detected on macOS")
    else:
        GPU_AVAILABLE = False
        GPU_DEVICE = 'cpu'
        print("🖥️  No GPU acceleration available, using CPU with PyTorch")
        
except ImportError:
    torch = None
    GPU_AVAILABLE = False
    GPU_DEVICE = 'cpu'
    print("⚠️  PyTorch not available. Install with: pip install torch")

class scICE:
    """
    Single-cell Inconsistency-based Clustering Ensemble (scICE) implementation in Python
    with GPU acceleration and progress bars
    """
    
    def __init__(self, n_jobs: int = -1, random_state: int = 42, use_gpu: bool = True, gpu_memory_fraction: float = 0.8):
        """
        Initialize scICE
        
        Parameters:
        -----------
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        random_state : int
            Random seed for reproducibility
        use_gpu : bool
            Whether to use GPU acceleration if available
        gpu_memory_fraction : float
            Fraction of GPU memory to use (0.1 to 1.0)
        """
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.results_ = {}
        self.gpu_memory_fraction = gpu_memory_fraction
        
        # GPU configuration
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            try:
                if GPU_DEVICE == 'cuda':
                    torch.cuda.set_device(0)
                    # 设置GPU内存分配策略 (CUDA only)
                    if hasattr(torch.cuda, 'set_memory_fraction'):
                        torch.cuda.set_memory_fraction(gpu_memory_fraction)
                elif GPU_DEVICE == 'mps':
                    # MPS doesn't need device setting
                    pass
                
                self.device = 'gpu'
                self.gpu_device = GPU_DEVICE  # Store actual device type
                
                if GPU_DEVICE == 'cuda':
                    print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
                    print(f"📱 GPU memory limit: {gpu_memory_fraction*100:.0f}%")
                elif GPU_DEVICE == 'mps':
                    print(f"🍎 Using Apple MPS GPU")
                    print(f"📱 Metal acceleration enabled")
                
                # GPU缓存管理
                self._gpu_cache = {}
                self._cache_size_limit = 10  # 最多缓存10个计算结果
                
            except Exception as e:
                print(f"⚠️  GPU initialization failed: {e}")
                self.use_gpu = False
                self.device = 'cpu'
                self.gpu_device = 'cpu'
        else:
            self.device = 'cpu'
            self.gpu_device = 'cpu'
            if use_gpu and not GPU_AVAILABLE:
                print("⚠️  GPU requested but not available. Using CPU.")
        
        # Choose array library
        self.xp = torch if self.use_gpu else np
        
        print(f"🔧 Device: {self.device.upper()}, Parallel jobs: {n_jobs}")
        
    def _clear_gpu_cache(self):
        """清理GPU缓存"""
        if self.use_gpu:
            self._gpu_cache.clear()
            if self.gpu_device == 'cuda':
                torch.cuda.empty_cache()
            elif self.gpu_device == 'mps':
                torch.mps.empty_cache()
            print("🧹 GPU cache cleared")
    
    def _get_cache_key(self, patterns_batch):
        """生成缓存键"""
        # 使用模式的哈希作为缓存键
        patterns_str = str([tuple(p) for p in patterns_batch])
        return hash(patterns_str)
    
    def _similarity_matrix_v2_batch_gpu_cached(self, patterns_batch: List[np.ndarray], d: float = 0.9):
        """
        带缓存的批量GPU相似度计算
        """
        if not self.use_gpu or len(patterns_batch) < 2:
            return self._similarity_matrix_batch_cpu(patterns_batch, d)
        
        # 检查缓存
        cache_key = self._get_cache_key(patterns_batch)
        if cache_key in self._gpu_cache:
            return self._gpu_cache[cache_key].clone()  # 返回副本避免修改原数据
        
        try:
            # 批量GPU计算
            result = self._similarity_matrix_v2_batch_gpu(patterns_batch, d)
            
            # 缓存结果（如果缓存未满）
            if len(self._gpu_cache) < self._cache_size_limit:
                self._gpu_cache[cache_key] = result.clone()
            
            return result
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️  GPU out of memory, clearing cache and falling back to CPU")
                self._clear_gpu_cache()
                return self._similarity_matrix_batch_cpu(patterns_batch, d)
            else:
                raise e
    
    def _to_gpu(self, array):
        """Convert array to GPU if GPU is being used"""
        if self.use_gpu and torch is not None:
            if isinstance(array, torch.Tensor):
                return array.to(self.gpu_device).float()  # Ensure float32
            else:
                return torch.tensor(array, dtype=torch.float32, device=self.gpu_device)
        return np.asarray(array, dtype=np.float32)
    
    def _to_cpu(self, array):
        """Convert array back to CPU"""
        if self.use_gpu and torch is not None and isinstance(array, torch.Tensor):
            return array.cpu().numpy().astype(np.float32)
        return np.asarray(array, dtype=np.float32)
        
    def _check_python_packages(self):
        """Check if required Python packages are installed"""
        try:
            import igraph
            import joblib
            import tqdm
            print("All required Python packages are properly installed.")
            
            if self.use_gpu:
                print(f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
                if GPU_AVAILABLE:
                    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            return True
        except ImportError as e:
            print(f"Missing required package: {e}")
            print("Please install: pip install igraph joblib tqdm")
            if self.use_gpu:
                print("For GPU support: pip install torch")
            return False
    
    def _graph_to_igraph(self, adj_matrix, weighted: bool = True):
        """
        Convert adjacency matrix to igraph object
        
        Parameters:
        -----------
        adj_matrix : scipy.sparse matrix
            Adjacency matrix
        weighted : bool
            Whether the graph is weighted
        
        Returns:
        --------
        igraph.Graph
        """
        import igraph as ig
        if not isinstance(adj_matrix, coo_matrix):
            adj_matrix = coo_matrix(adj_matrix)
        
        edges = list(zip(adj_matrix.row, adj_matrix.col))
        weights = adj_matrix.data if weighted else None
        
        g = ig.Graph(n=adj_matrix.shape[0], edges=edges, directed=False)
        if weighted and weights is not None:
            g.es['weight'] = weights
            
        return g
    
    def _cluster_graph(self, g, gamma: float = 0.8, 
                      objective_function: str = "CPM", n_iter: int = 5, 
                      beta: float = 0.1, init_membership: Optional[List] = None):
        """
        Cluster graph using Leiden algorithm
        
        Parameters:
        -----------
        g : igraph.Graph
            Input graph
        gamma : float
            Resolution parameter
        objective_function : str
            Objective function for clustering ('CPM' or 'modularity')
        n_iter : int
            Number of iterations
        beta : float
            Beta parameter for Leiden
        init_membership : list, optional
            Initial membership assignment
        
        Returns:
        --------
        np.ndarray
            Cluster membership
        """
        if g.is_weighted():
            partition = g.community_leiden(
                resolution=gamma,
                weights='weight',
                objective_function=objective_function,
                n_iterations=n_iter,
                beta=beta,
                initial_membership=init_membership
            )
        else:
            partition = g.community_leiden(
                resolution=gamma,
                objective_function=objective_function,
                n_iterations=n_iter,
                beta=beta,
                initial_membership=init_membership
            )
        
        return np.array(partition.membership, dtype=np.int16)
    
    def _extract_array(self, cluster_matrix):
        """
        Extract unique clustering patterns and their probabilities
        
        Parameters:
        -----------
        cluster_matrix : np.ndarray
            Matrix of clustering results
        
        Returns:
        --------
        dict
            Dictionary with 'arr' (unique patterns) and 'parr' (probabilities)
        """
        # Convert each row to tuple for hashing
        patterns = [tuple(row) for row in cluster_matrix]
        pattern_counts = Counter(patterns)
        
        # Sort by frequency
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        
        unique_patterns = [np.array(pattern) for pattern, _ in sorted_patterns]
        counts = [count for _, count in sorted_patterns]
        probabilities = np.array(counts) / np.sum(counts)
        
        return {'arr': unique_patterns, 'parr': probabilities}
    
    def _similarity_matrix_v2(self, cluster_a: np.ndarray, cluster_b: np.ndarray, 
                             d: float = 0.9, return_vector: bool = True):
        """
        Compute similarity between two clustering solutions with GPU acceleration
        
        Parameters:
        -----------
        cluster_a, cluster_b : np.ndarray
            Clustering assignments
        d : float
            Damping parameter
        return_vector : bool
            If True, return mean similarity; if False, return vector
        
        Returns:
        --------
        float or np.ndarray
            Similarity score(s)
        """
        # Move to GPU if available
        if self.use_gpu:
            cluster_a_gpu = torch.tensor(cluster_a, dtype=torch.float32, device=self.gpu_device)
            cluster_b_gpu = torch.tensor(cluster_b, dtype=torch.float32, device=self.gpu_device)
            n = len(cluster_a_gpu)
            unique_a = torch.unique(cluster_a_gpu)
            unique_b = torch.unique(cluster_b_gpu)
        else:
            n = len(cluster_a)
            unique_a = np.unique(cluster_a)
            unique_b = np.unique(cluster_b)
        
        # Build cluster membership lists
        if self.use_gpu:
            clusters_a = {int(cluster): torch.where(cluster_a_gpu == cluster)[0] for cluster in unique_a}
            clusters_b = {int(cluster): torch.where(cluster_b_gpu == cluster)[0] for cluster in unique_b}
        else:
            clusters_a = {cluster: np.where(cluster_a == cluster)[0] for cluster in unique_a}
            clusters_b = {cluster: np.where(cluster_b == cluster)[0] for cluster in unique_b}
        
        # Compute cluster sizes and probabilities
        size_a = {cluster: d / len(members) for cluster, members in clusters_a.items()}
        size_b = {cluster: d / len(members) for cluster, members in clusters_b.items()}
        
        if self.use_gpu:
            ecs = torch.zeros(n, device=self.gpu_device, dtype=torch.float32)
            ppr1 = torch.zeros(n, device=self.gpu_device, dtype=torch.float32)
            ppr2 = torch.zeros(n, device=self.gpu_device, dtype=torch.float32)
        else:
            ecs = np.zeros(n, dtype=np.float32)
            ppr1 = np.zeros(n, dtype=np.float32)
            ppr2 = np.zeros(n, dtype=np.float32)
        
        # Cache for computed similarities
        similarity_cache = {}
        
        for i in range(n):
            if self.use_gpu:
                cluster_i_a = int(cluster_a_gpu[i])
                cluster_i_b = int(cluster_b_gpu[i])
            else:
                cluster_i_a = cluster_a[i]
                cluster_i_b = cluster_b[i]
            
            cache_key = (cluster_i_a, cluster_i_b)
            if cache_key in similarity_cache:
                ecs[i] = similarity_cache[cache_key]
                continue
            
            # Get cluster members
            members_a = clusters_a[cluster_i_a]
            members_b = clusters_b[cluster_i_b]
            
            if self.use_gpu:
                # Use torch.cat and torch.unique for union operation
                all_members = torch.unique(torch.cat([members_a, members_b]))
            else:
                all_members = np.union1d(members_a, members_b)
            
            # Reset probability vectors
            if self.use_gpu:
                ppr1.fill_(0)
                ppr2.fill_(0)
            else:
                ppr1.fill(0)
                ppr2.fill(0)
            
            # Set probabilities for cluster A
            ppr1[members_a] = size_a[cluster_i_a]
            ppr1[i] = 1.0 - d + size_a[cluster_i_a]
            
            # Set probabilities for cluster B
            ppr2[members_b] = size_b[cluster_i_b]
            ppr2[i] = 1.0 - d + size_b[cluster_i_b]
            
            # Compute Earth Mover's Distance
            if self.use_gpu:
                earth_score = torch.sum(torch.abs(ppr2[all_members] - ppr1[all_members]))
            else:
                earth_score = np.sum(np.abs(ppr2[all_members] - ppr1[all_members]))
            
            ecs[i] = earth_score
            similarity_cache[cache_key] = float(earth_score)
        
        similarities = 1 - (1 / (2 * d)) * ecs
        
        if return_vector:
            if self.use_gpu:
                return float(torch.mean(similarities))
            else:
                return np.mean(similarities)
        else:
            return self._to_cpu(similarities)
    
    def _get_ic2(self, extracted_data: Dict):
        """
        Compute Inconsistency Index (IC) - optimized with cached batch GPU processing
        """
        patterns = extracted_data['arr']
        probabilities = extracted_data['parr']
        
        if probabilities.sum() != 1:
            probabilities = probabilities / probabilities.sum()
        
        n_patterns = len(patterns)
        
        if n_patterns == 1:
            return patterns, np.array([[1.0]]), probabilities, 1.0
        
        # 批量GPU计算相似度矩阵（带缓存）
        if self.use_gpu and n_patterns > 2:
            try:
                # 使用缓存的批量GPU计算
                S_ab_gpu = self._similarity_matrix_v2_batch_gpu_cached(patterns)
                S_ab = self._to_cpu(S_ab_gpu)
                
                # GPU上计算IC分数
                probabilities_gpu = torch.tensor(probabilities, device=self.gpu_device, dtype=torch.float32)
                ic_score = float(torch.dot(torch.mv(S_ab_gpu, probabilities_gpu), probabilities_gpu))
                
            except Exception as e:
                print(f"GPU batch computation failed, falling back to CPU: {e}")
                # 回退到CPU计算
                S_ab = self._similarity_matrix_batch_cpu(patterns)
                ic_score = np.dot(np.dot(S_ab, probabilities), probabilities)
        else:
            # CPU计算
            S_ab = self._similarity_matrix_batch_cpu(patterns)
            ic_score = np.dot(np.dot(S_ab, probabilities), probabilities)
        
        return np.array(patterns), S_ab, probabilities, ic_score
    
    def _get_mei_from_array(self, extracted_data: Dict):
        """
        Compute Mean Element-wise Inconsistency (MEI) - optimized with batch GPU processing
        """
        patterns = extracted_data['arr']
        probabilities = extracted_data['parr']
        
        if len(patterns) == 1:
            return np.ones(len(patterns[0]), dtype=np.float32)
        
        n_patterns = len(patterns)
        n_cells = len(patterns[0])
        
        # 使用批量GPU计算或CPU回退
        if self.use_gpu and n_patterns > 2:
            try:
                # 批量GPU计算相似度矩阵
                similarity_matrix_gpu = self._similarity_matrix_v2_batch_gpu_cached(patterns)
                
                # 在GPU上计算加权相似度
                probabilities_gpu = torch.tensor(probabilities, device=self.gpu_device, dtype=torch.float32)
                
                # 计算所有配对的加权相似度并求和
                total_similarities = torch.zeros(n_cells, device=self.gpu_device, dtype=torch.float32)
                n_pairs = 0
                
                for i in range(n_patterns):
                    for j in range(i+1, n_patterns):
                        # 获取单个相似度向量（需要重新计算，因为矩阵只存储标量值）
                        sim_vector = self._similarity_matrix_v2(patterns[i], patterns[j], return_vector=False)
                        sim_vector_gpu = torch.tensor(sim_vector, device=self.gpu_device, dtype=torch.float32)
                        weight = probabilities[i] + probabilities[j]
                        total_similarities += sim_vector_gpu * weight
                        n_pairs += 1
                
                # 归一化
                if n_pairs > 0:
                    result = total_similarities / n_pairs
                else:
                    result = torch.ones(n_cells, device=self.gpu_device, dtype=torch.float32)
                
                return self._to_cpu(result)
                
            except Exception as e:
                print(f"GPU MEI computation failed, falling back to CPU: {e}")
                # 回退到CPU计算
                pass
        
        # CPU计算（回退方案）
        total_similarities = np.zeros(n_cells, dtype=np.float32)
        pairs = [(i, j) for i in range(n_patterns) for j in range(i+1, n_patterns)]
        
        for i, j in pairs:
            sim_vector = self._similarity_matrix_v2(patterns[i], patterns[j], return_vector=False)
            weight = probabilities[i] + probabilities[j]
            total_similarities += np.asarray(sim_vector, dtype=np.float32) * weight
        
        # 归一化
        n_pairs = len(pairs)
        if n_pairs > 0:
            result = total_similarities / n_pairs
        else:
            result = np.ones(n_cells, dtype=np.float32)
        
        return result
    
    def _get_best_labels(self, extracted_data: Dict):
        """
        Get the best clustering labels - optimized with cached batch processing
        """
        patterns = extracted_data['arr']
        n_patterns = len(patterns)
        
        if n_patterns == 1:
            return patterns[0]
        
        # 批量计算相似度矩阵（带缓存）
        if self.use_gpu and n_patterns > 2:
            try:
                # 使用缓存的批量GPU计算
                S_full_gpu = self._similarity_matrix_v2_batch_gpu_cached(patterns)
                total_similarities = torch.sum(S_full_gpu, dim=1)
                best_idx = int(torch.argmax(total_similarities))
                return patterns[best_idx]
                
            except Exception as e:
                print(f"GPU best labels computation failed, falling back to CPU: {e}")
                # 回退到CPU
                pass
        
        # CPU计算（回退方案）
        S_full = self._similarity_matrix_batch_cpu(patterns)
        total_similarities = np.sum(S_full, axis=1)
        best_idx = np.argmax(total_similarities)
        
        return patterns[best_idx]
    
    def fit(self, adata: ad.AnnData, use_rep: str = 'X_pca', 
           graph_type: str = 'umap', resolution_range: Tuple[int, int] = (1, 20),
           n_steps: int = 11, n_trials: int = 15, n_boot: int = 100,
           beta: float = 0.1, n_clusters: int = 10, delta_n: int = 2,
           max_iter: int = 150, remove_threshold: float = 1.15,
           objective_function: str = "CPM", val_tolerance: float = 1e-8):
        """
        Fit scICE clustering to single-cell data
        
        Parameters:
        -----------
        adata : anndata.AnnData
            Annotated data object
        use_rep : str
            Representation to use for clustering
        graph_type : str
            Type of graph to use ('umap', 'snn', 'knn')
        resolution_range : tuple
            Range of cluster numbers to explore
        n_steps : int
            Number of steps in resolution search
        n_trials : int
            Number of clustering trials per resolution
        n_boot : int
            Number of bootstrap samples for IC estimation
        beta : float
            Beta parameter for Leiden clustering
        n_clusters : int
            Initial number of clusters for pre-processing
        delta_n : int
            Step size for iteration refinement
        max_iter : int
            Maximum number of iterations
        remove_threshold : float
            Threshold for removing unstable solutions
        objective_function : str
            Objective function ('CPM' or 'modularity')
        val_tolerance : float
            Convergence tolerance
        
        Returns:
        --------
        self
        """
        if not self._check_python_packages():
            return self
        
        # Ensure graph is built
        print("Building neighborhood graph...")
        from ..pp import neighbors as _neighbors
        from ._compat import umap_like_scanpy as _umap
        if graph_type == 'umap':
            if 'connectivities' not in adata.obsp:
                _neighbors(adata, use_rep=use_rep, random_state=self.random_state)
            if 'X_umap' not in adata.obsm:
                _umap(adata, random_state=self.random_state)
            graph = adata.obsp['connectivities']
        elif graph_type == 'snn':
            if 'connectivities' not in adata.obsp:
                _neighbors(adata, use_rep=use_rep, random_state=self.random_state)
            graph = adata.obsp['connectivities']
        elif graph_type == 'knn':
            if 'distances' not in adata.obsp:
                _neighbors(adata, use_rep=use_rep, random_state=self.random_state)
            graph = adata.obsp['distances']
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")
        
        # Convert to igraph
        print("Converting graph to igraph format...")
        igraph_obj = self._graph_to_igraph(graph, weighted=True)
        
        # Main clustering procedure
        print(f"Starting scICE clustering with {self.device.upper()}...")
        self._clustering_procedure(
            igraph_obj, adata, resolution_range, n_steps, n_trials, n_boot,
            beta, n_clusters, delta_n, max_iter, remove_threshold,
            objective_function, val_tolerance
        )
        
        return self
    
    def _clustering_procedure(self, igraph_obj, adata, resolution_range, n_steps, 
                            n_trials, n_boot, beta, n_clusters, delta_n, max_iter,
                            remove_threshold, objective_function, val_tolerance):
        """
        Main clustering procedure implementation with optimized parallelization
        Now each thread processes one cluster number (target_k) for maximum speedup
        """
        t_range = list(range(resolution_range[0], resolution_range[1] + 1))
        
        # Resolution search bounds
        if objective_function == "modularity":
            start_g, end_g = 0, 10
        elif objective_function == "CPM":
            start_g, end_g = np.log(val_tolerance), 0
        
        print(f"Exploring {len(t_range)} cluster numbers: {t_range}")
        
        def process_single_cluster_number(target_k):
            """Process a single cluster number (target_k) completely"""
            try:
                # Binary search for appropriate resolution
                gamma_range = self._find_resolution_range(
                    igraph_obj, target_k, start_g, end_g, objective_function,
                    n_clusters, beta, val_tolerance
                )
                
                if gamma_range is None:
                    return None
                
                # Fine-tune clustering at this resolution
                best_result = self._optimize_clustering(
                    igraph_obj, target_k, gamma_range, n_steps, n_trials, 
                    n_boot, beta, delta_n, max_iter, objective_function
                )
                
                if best_result is not None:
                    best_result['target_k'] = target_k
                    return best_result
                else:
                    return None
                    
            except Exception as e:
                print(f"Error processing k={target_k}: {e}")
                return None
        
        # Choose processing strategy based on n_jobs
        if self.n_jobs == 1:
            print(f"🔧 Using sequential processing with real-time progress updates")
            # Sequential processing with real-time progress updates (like old version)
            successful_results = []
            
            try:
                with progress_bar(t_range, desc="Processing cluster numbers", 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                         leave=True) as pbar:
                    
                    for target_k in pbar:
                        pbar.set_description(f"Processing k={target_k}")
                        
                        result = process_single_cluster_number(target_k)
                        if result is not None:
                            successful_results.append(result)
                            # Update progress bar with current results
                            pbar.set_postfix({
                                'IC': f"{result['ic']:.4f}",
                                'γ': f"{result['gamma']:.4f}",
                                'Found': len(successful_results)
                            })
                            
            except KeyboardInterrupt:
                print("\nClustering interrupted by user")
            except Exception as e:
                print(f"\nError during sequential clustering: {e}")
        else:
            print(f"🚀 Using parallel processing: each thread handles one cluster number")
            # Parallel processing of all cluster numbers
            print(f"🔧 Processing {len(t_range)} cluster numbers in parallel...")
            
            try:
                with progress_bar(total=len(t_range), desc="Processing cluster numbers in parallel", 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                         leave=True) as pbar:
                    
                    # Use joblib to process cluster numbers in parallel
                    results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                        delayed(process_single_cluster_number)(target_k) 
                        for target_k in t_range
                    )
                    
                    # Update progress bar
                    pbar.update(len(t_range))
                    
            except KeyboardInterrupt:
                print("\nClustering interrupted by user")
                results = []
            except Exception as e:
                print(f"\nError during parallel clustering: {e}")
                results = []
            
            # Filter out None results
            successful_results = [r for r in results if r is not None]
        
        # Organize output (common for both strategies)
        if successful_results:
            # Sort by target_k to maintain order
            successful_results.sort(key=lambda x: x['target_k'])
            
            # Extract results
            list_gamma = [r['gamma'] for r in successful_results]
            list_labels = [r['labels'] for r in successful_results]
            list_ic = [r['ic'] for r in successful_results]
            list_ic_vec = [r['ic_vec'] for r in successful_results]
            list_best_labels = [r['best_labels'] for r in successful_results]
            list_n_clusters = [r['target_k'] for r in successful_results]
            list_n_iter = [r['n_iter'] for r in successful_results]
            
            # Store results
            self.results_ = {
                'gamma': list_gamma,
                'labels': list_labels,
                'ic': list_ic,
                'ic_vec': list_ic_vec,
                'best_labels': list_best_labels,
                'n_clusters': list_n_clusters,
                'n_iter': list_n_iter,
                'adata': adata,
                'graph': igraph_obj
            }
            
            # Compute MEI scores with progress bar
            print("Computing MEI scores...")
            try:
                with progress_bar(list_labels, desc="Computing MEI", leave=False) as mei_pbar:
                    mei_scores = []
                    for labels in mei_pbar:
                        mei_scores.append(self._get_mei_from_array(labels))
                self.results_['mei'] = mei_scores
            except Exception as e:
                print(f"Error computing MEI scores: {e}")
                self.results_['mei'] = []
        else:
            # No successful results
            self.results_ = {
                'gamma': [], 'labels': [], 'ic': [], 'ic_vec': [],
                'best_labels': [], 'n_clusters': [], 'n_iter': [],
                'mei': [], 'adata': adata, 'graph': igraph_obj
            }
        
        n_found = len(successful_results)
        print(f"\n✅ Completed scICE clustering. Found {n_found}/{len(t_range)} stable solutions.")
        
        if self.use_gpu:
            if self.gpu_device == 'cuda':
                print(f"📊 GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            elif self.gpu_device == 'mps':
                print(f"📊 MPS GPU memory in use")
            print(f"🗂️  GPU cache entries: {len(getattr(self, '_gpu_cache', {}))}")
            # 清理GPU缓存
            self._clear_gpu_cache()
        
        # Performance summary
        if n_found > 0:
            avg_ic = np.mean([r['ic'] for r in successful_results])
            print(f"📊 Average IC score: {avg_ic:.4f}")
            print(f"🎯 Stable cluster numbers found: {sorted([r['target_k'] for r in successful_results])}")
        else:
            print("⚠️  No stable solutions found. Try:")
            print("   • Increasing resolution range")
            print("   • Adjusting IC threshold") 
            print("   • Using different objective function")
    
    def _find_resolution_range(self, igraph_obj, target_k, start_g, end_g, 
                              objective_function, n_clusters, beta, val_tolerance):
        """Find resolution range that produces target number of clusters"""
        
        def get_median_clusters(gamma):
            clusterings = Parallel(n_jobs=self.n_jobs)(
                delayed(self._cluster_graph)(
                    igraph_obj, gamma=gamma, objective_function=objective_function,
                    n_iter=3, beta=0.01
                ) for _ in range(n_clusters)
            )
            cluster_counts = [np.max(clustering) + 1 for clustering in clusterings]
            return np.median(cluster_counts)
        
        left, right = start_g, end_g
        
        # Binary search for resolution range
        max_iterations = 20
        for _ in range(max_iterations):
            if objective_function == "CPM" and abs(np.exp(left) - np.exp(right)) < val_tolerance:
                break
            elif objective_function == "modularity" and abs(left - right) < val_tolerance:
                break
            
            mid = (left + right) / 2
            gamma = np.exp(mid) if objective_function == "CPM" else mid
            
            median_k = get_median_clusters(gamma)
            
            if median_k < target_k:
                left = mid
            else:
                right = mid
        
        if objective_function == "CPM":
            return (np.exp(left), np.exp(right))
        else:
            return (left, right)
    
    def _optimize_clustering(self, igraph_obj, target_k, gamma_range, n_steps,
                           n_trials, n_boot, beta, delta_n, max_iter, objective_function):
        """Optimize clustering for a specific target cluster number (thread-safe version)"""
        
        # Create gamma grid
        if objective_function == "CPM":
            gamma_grid = np.logspace(np.log10(gamma_range[0]), np.log10(gamma_range[1]), n_steps)
        else:
            gamma_grid = np.linspace(gamma_range[0], gamma_range[1], n_steps)
        
        best_ic = float('inf')
        best_result = None
        
        # Optimize without nested progress bars (since we're running in parallel)
        for gamma in gamma_grid:
            try:
                # Generate multiple clusterings
                clusterings = Parallel(n_jobs=1, verbose=0)(  # Use single job per gamma to avoid oversubscription
                    delayed(self._cluster_graph)(
                        igraph_obj, gamma=gamma, objective_function=objective_function,
                        n_iter=10, beta=beta
                    ) for _ in range(n_trials)
                )
                
                clustering_matrix = np.array(clusterings)
                cluster_counts = np.max(clustering_matrix, axis=1) + 1
                
                # Filter for target cluster number
                valid_mask = cluster_counts == target_k
                if not np.any(valid_mask):
                    continue
                
                valid_clusterings = clustering_matrix[valid_mask]
                
                # Extract patterns and compute IC
                extracted_data = self._extract_array(valid_clusterings)
                _, _, _, ic_score = self._get_ic2(extracted_data)
                
                # Bootstrap IC estimation (simplified for parallel execution)
                ic_bootstrap = []
                for _ in range(n_boot):
                    try:
                        boot_indices = np.random.choice(len(valid_clusterings), 
                                                       size=len(valid_clusterings), replace=True)
                        boot_data = self._extract_array(valid_clusterings[boot_indices])
                        _, _, _, boot_ic = self._get_ic2(boot_data)
                        ic_bootstrap.append(1.0 / boot_ic)
                    except Exception:
                        ic_bootstrap.append(1.0 / ic_score)  # Fallback
                
                median_ic = np.median(ic_bootstrap) if ic_bootstrap else 1.0 / ic_score
                
                if median_ic < best_ic:
                    best_ic = median_ic
                    best_labels = self._get_best_labels(extracted_data)
                    best_result = {
                        'gamma': gamma,
                        'labels': extracted_data,
                        'ic': median_ic,
                        'ic_vec': ic_bootstrap,
                        'best_labels': best_labels,
                        'n_iter': 10  # Base iterations
                    }
                    
            except Exception as e:
                # Silent failure for individual gamma values to avoid spam in parallel execution
                continue
        
        return best_result
    
    def _find_resolution_range_single(self, igraph_obj, target_k, start_g, end_g, 
                                     objective_function, n_clusters, beta, val_tolerance):
        """
        Find resolution range for a single cluster number (thread-safe version)
        """
        def get_median_clusters(gamma):
            # Use sequential processing to avoid nested parallelization
            clusterings = []
            for _ in range(n_clusters):
                clustering = self._cluster_graph(
                    igraph_obj, gamma=gamma, objective_function=objective_function,
                    n_iter=3, beta=0.01
                )
                clusterings.append(clustering)
            cluster_counts = [np.max(clustering) + 1 for clustering in clusterings]
            return np.median(cluster_counts)
        
        left, right = start_g, end_g
        
        # Binary search for resolution range
        max_iterations = 20
        for _ in range(max_iterations):
            if objective_function == "CPM" and abs(np.exp(left) - np.exp(right)) < val_tolerance:
                break
            elif objective_function == "modularity" and abs(left - right) < val_tolerance:
                break
            
            mid = (left + right) / 2
            gamma = np.exp(mid) if objective_function == "CPM" else mid
            
            median_k = get_median_clusters(gamma)
            
            if median_k < target_k:
                left = mid
            else:
                right = mid
        
        if objective_function == "CPM":
            return (np.exp(left), np.exp(right))
        else:
            return (left, right)
    
    def _optimize_clustering_single(self, igraph_obj, target_k, gamma_range, n_steps,
                                   n_trials, n_boot, beta, delta_n, max_iter, objective_function):
        """
        Optimize clustering for a single cluster number (thread-safe version)
        """
        # Create gamma grid
        if objective_function == "CPM":
            gamma_grid = np.logspace(np.log10(gamma_range[0]), np.log10(gamma_range[1]), n_steps)
        else:
            gamma_grid = np.linspace(gamma_range[0], gamma_range[1], n_steps)
        
        best_ic = float('inf')
        best_result = None
        
        # Process gamma values sequentially (no nested parallelization)
        for gamma in gamma_grid:
            # Generate multiple clusterings sequentially
            clusterings = []
            for _ in range(n_trials):
                clustering = self._cluster_graph(
                    igraph_obj, gamma=gamma, objective_function=objective_function,
                    n_iter=10, beta=beta
                )
                clusterings.append(clustering)
            
            clustering_matrix = np.array(clusterings)
            cluster_counts = np.max(clustering_matrix, axis=1) + 1
            
            # Filter for target cluster number
            valid_mask = cluster_counts == target_k
            if not np.any(valid_mask):
                continue
            
            valid_clusterings = clustering_matrix[valid_mask]
            
            # Extract patterns and compute IC
            extracted_data = self._extract_array(valid_clusterings)
            _, _, _, ic_score = self._get_ic2_single(extracted_data)
            
            # Bootstrap IC estimation (sequential)
            ic_bootstrap = []
            try:
                for _ in range(n_boot):
                    boot_indices = np.random.choice(len(valid_clusterings), 
                                                   size=len(valid_clusterings), replace=True)
                    boot_data = self._extract_array(valid_clusterings[boot_indices])
                    _, _, _, boot_ic = self._get_ic2_single(boot_data)
                    ic_bootstrap.append(1.0 / boot_ic)
            except Exception as e:
                ic_bootstrap = [1.0 / ic_score]  # Fallback
            
            median_ic = np.median(ic_bootstrap)
            
            if median_ic < best_ic:
                best_ic = median_ic
                best_labels = self._get_best_labels_single(extracted_data)
                best_result = {
                    'gamma': gamma,
                    'labels': extracted_data,
                    'ic': median_ic,
                    'ic_vec': ic_bootstrap,
                    'best_labels': best_labels,
                    'n_iter': 10
                }
        
        return best_result
    
    def _get_ic2_single(self, extracted_data):
        """
        Compute IC for single cluster number (thread-safe version)
        """
        patterns = np.array(extracted_data['arr'])
        probabilities = extracted_data['parr']
        
        if probabilities.sum() != 1:
            probabilities = probabilities / probabilities.sum()
        
        n_patterns = len(patterns)
        
        # Compute pairwise similarities sequentially
        def compute_sim(i, j):
            return self._similarity_matrix_v2(patterns[i], patterns[j])
        
        pairs = [(i, j) for i in range(n_patterns) for j in range(i+1, n_patterns)]
        similarities = []
        for i, j in pairs:
            similarities.append(compute_sim(i, j))
        
        # Build similarity matrix
        if self.use_gpu:
            S_ab = torch.eye(n_patterns, device='cuda', dtype=torch.float32)
            probabilities_gpu = torch.tensor(probabilities, device='cuda', dtype=torch.float32)
        else:
            S_ab = np.eye(n_patterns, dtype=np.float32)
        
        idx = 0
        for i in range(n_patterns):
            for j in range(i+1, n_patterns):
                S_ab[i, j] = similarities[idx]
                S_ab[j, i] = similarities[idx]
                idx += 1
        
        # Compute IC score
        if self.use_gpu:
            ic_score = float(torch.dot(torch.mv(S_ab, probabilities_gpu), probabilities_gpu))
            S_ab = self._to_cpu(S_ab)
        else:
            ic_score = np.dot(np.dot(S_ab, probabilities), probabilities)
        
        return patterns, S_ab, probabilities, ic_score
    
    def _get_best_labels_single(self, extracted_data):
        """
        Get best labels for single cluster number (thread-safe version)
        """
        patterns = np.array(extracted_data['arr'])
        n_patterns = len(patterns)
        
        if n_patterns == 1:
            return patterns[0]
        
        # Compute similarities sequentially
        def compute_sim_pair(i, j):
            return self._similarity_matrix_v2(patterns[i], patterns[j])
        
        pairs = [(i, j) for i in range(n_patterns) for j in range(n_patterns) if i != j]
        similarities = []
        for i, j in pairs:
            similarities.append(compute_sim_pair(i, j))
        
        # Build full similarity matrix
        if self.use_gpu:
            S_full = torch.eye(n_patterns, device='cuda', dtype=torch.float32)
        else:
            S_full = np.eye(n_patterns, dtype=np.float32)
        
        idx = 0
        for i in range(n_patterns):
            for j in range(n_patterns):
                if i != j:
                    S_full[i, j] = similarities[idx]
                    idx += 1
        
        # Find pattern with highest total similarity
        if self.use_gpu:
            total_similarities = torch.sum(S_full, dim=1)
            best_idx = int(torch.argmax(total_similarities))
        else:
            total_similarities = np.sum(S_full, axis=1)
            best_idx = np.argmax(total_similarities)
        
        return patterns[best_idx]
    
    def get_stable_labels(self, threshold: float = 1.005) -> pd.DataFrame:
        """
        Get stable clustering labels based on IC threshold
        
        Parameters:
        -----------
        threshold : float
            IC threshold for stable solutions
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with stable clustering labels
        """
        if not self.results_:
            raise ValueError("No clustering results found. Run fit() first.")
        
        # Filter stable solutions
        stable_mask = np.array(self.results_['ic']) < threshold
        stable_k = np.array(self.results_['n_clusters'])[stable_mask]
        stable_labels = np.array(self.results_['best_labels'])[stable_mask]
        
        # Create DataFrame
        n_cells = len(self.results_['adata'].obs_names)
        result_df = pd.DataFrame(index=self.results_['adata'].obs_names)
        
        for i, k in enumerate(stable_k):
            result_df[f'scICE_k{k}'] = stable_labels[i]
        
        return result_df
    
    def plot_ic(self, threshold: float = 1.005, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot Inconsistency Index across cluster numbers
        
        Parameters:
        -----------
        threshold : float
            IC threshold line to show
        figsize : tuple
            Figure size
        
        Returns:
        --------
        matplotlib.figure.Figure
        """
        if not self.results_:
            raise ValueError("No clustering results found. Run fit() first.")
        
        # Prepare data for plotting
        x_data = []
        y_data = []
        
        for i, k in enumerate(self.results_['n_clusters']):
            ic_vec = self.results_['ic_vec'][i]
            x_data.extend([k] * len(ic_vec))
            y_data.extend(ic_vec)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Box plot
        unique_k = sorted(set(x_data))
        box_data = [y_data[i] for i, k in enumerate(x_data) if k in unique_k]
        
        # Group data by k
        grouped_data = {}
        for x, y in zip(x_data, y_data):
            if x not in grouped_data:
                grouped_data[x] = []
            grouped_data[x].append(y)
        
        positions = list(grouped_data.keys())
        box_data = [grouped_data[k] for k in positions]
        
        ax.boxplot(box_data, positions=positions)
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold = {threshold}')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('IC (Inconsistency Index)')
        ax.set_title('scICE Clustering Stability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add device info
        device_text = f"Computed on: {self.device.upper()}"
        ax.text(0.02, 0.98, device_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def add_to_adata(self, adata: Optional[ad.AnnData] = None, threshold: float = 1.005):
        """
        Add clustering results to AnnData object
        
        Parameters:
        -----------
        adata : anndata.AnnData, optional
            AnnData object to add results to. If None, uses the one from fitting.
        threshold : float
            IC threshold for stable solutions
        """
        if adata is None:
            adata = self.results_['adata']
        
        stable_df = self.get_stable_labels(threshold=threshold)
        
        # Add to adata.obs
        for col in stable_df.columns:
            adata.obs[col] = stable_df[col].astype('category')
        
        print(f"Added {len(stable_df.columns)} stable clustering solutions to adata.obs")
        
    def get_memory_usage(self):
        """Get current memory usage information including GPU cache"""
        if self.use_gpu and torch is not None:
            cache_entries = len(getattr(self, '_gpu_cache', {}))
            
            if self.gpu_device == 'cuda':
                device_props = torch.cuda.get_device_properties(0)
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                total = device_props.total_memory / 1e9
                
                return {
                    'device': 'GPU',
                    'gpu_type': 'CUDA',
                    'used_memory_gb': allocated,
                    'reserved_memory_gb': reserved,
                    'total_memory_gb': total,
                    'device_name': device_props.name,
                    'cache_entries': cache_entries,
                    'memory_fraction': getattr(self, 'gpu_memory_fraction', 1.0)
                }
            elif self.gpu_device == 'mps':
                return {
                    'device': 'GPU',
                    'gpu_type': 'MPS',
                    'used_memory_gb': 0.0,  # MPS doesn't provide detailed memory info
                    'reserved_memory_gb': 0.0,
                    'total_memory_gb': 0.0,
                    'device_name': 'Apple Metal GPU',
                    'cache_entries': cache_entries,
                    'memory_fraction': getattr(self, 'gpu_memory_fraction', 1.0)
                }
        else:
            import psutil
            return {
                'device': 'CPU',
                'used_memory_gb': psutil.virtual_memory().used / 1e9,
                'total_memory_gb': psutil.virtual_memory().total / 1e9,
                'cpu_count': psutil.cpu_count()
            }
    
    def _similarity_matrix_v2_batch_gpu(self, patterns_batch: List[np.ndarray], d: float = 0.9):
        """
        批量GPU相似度计算 - 最小化CPU-GPU数据传输
        
        Parameters:
        -----------
        patterns_batch : List[np.ndarray]
            多个聚类模式的批次
        d : float
            阻尼参数
            
        Returns:
        --------
        torch.Tensor
            相似度矩阵 (GPU上)
        """
        if not self.use_gpu or len(patterns_batch) < 2:
            # 回退到CPU计算
            return self._similarity_matrix_batch_cpu(patterns_batch, d)
        
        # 将所有模式一次性传输到GPU
        n_patterns = len(patterns_batch)
        n_cells = len(patterns_batch[0])
        
        # 批量传输到GPU
        patterns_gpu = torch.tensor(
            np.array(patterns_batch), 
            dtype=torch.float32, 
            device=self.gpu_device
        )  # Shape: (n_patterns, n_cells)
        
        # 在GPU上批量计算所有相似度
        similarity_matrix = torch.zeros(
            (n_patterns, n_patterns), 
            dtype=torch.float32, 
            device=self.gpu_device
        )
        
        # 并行计算所有成对相似度
        for i in range(n_patterns):
            for j in range(i, n_patterns):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self._compute_similarity_gpu_vectorized(
                        patterns_gpu[i], patterns_gpu[j], d
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        return similarity_matrix
    
    def _compute_similarity_gpu_vectorized(self, pattern_a_gpu: torch.Tensor, 
                                         pattern_b_gpu: torch.Tensor, d: float = 0.9):
        """
        GPU向量化相似度计算 - 全部在GPU上完成
        """
        n = pattern_a_gpu.shape[0]
        
        # 获取唯一聚类标签
        unique_a = torch.unique(pattern_a_gpu)
        unique_b = torch.unique(pattern_b_gpu)
        
        # 构建聚类成员掩码（向量化）
        clusters_mask_a = []
        clusters_mask_b = []
        cluster_sizes_a = []
        cluster_sizes_b = []
        
        for cluster in unique_a:
            mask = (pattern_a_gpu == cluster)
            clusters_mask_a.append(mask)
            cluster_sizes_a.append(d / mask.sum().float())
        
        for cluster in unique_b:
            mask = (pattern_b_gpu == cluster)
            clusters_mask_b.append(mask)
            cluster_sizes_b.append(d / mask.sum().float())
        
        # 向量化计算相似度
        ecs = torch.zeros(n, dtype=torch.float32, device=self.gpu_device)
        
        for i in range(n):
            # 找到当前细胞属于哪个聚类
            cluster_idx_a = None
            cluster_idx_b = None
            
            for idx, mask in enumerate(clusters_mask_a):
                if mask[i]:
                    cluster_idx_a = idx
                    break
            
            for idx, mask in enumerate(clusters_mask_b):
                if mask[i]:
                    cluster_idx_b = idx
                    break
            
            if cluster_idx_a is not None and cluster_idx_b is not None:
                # 获取相关细胞
                mask_a = clusters_mask_a[cluster_idx_a]
                mask_b = clusters_mask_b[cluster_idx_b]
                all_cells_mask = mask_a | mask_b
                
                # 构建概率向量
                ppr1 = torch.zeros(n, dtype=torch.float32, device=self.gpu_device)
                ppr2 = torch.zeros(n, dtype=torch.float32, device=self.gpu_device)
                
                ppr1[mask_a] = cluster_sizes_a[cluster_idx_a]
                ppr1[i] = 1.0 - d + cluster_sizes_a[cluster_idx_a]
                
                ppr2[mask_b] = cluster_sizes_b[cluster_idx_b]
                ppr2[i] = 1.0 - d + cluster_sizes_b[cluster_idx_b]
                
                # 计算Earth Mover's Distance (向量化)
                ecs[i] = torch.sum(torch.abs(ppr2[all_cells_mask] - ppr1[all_cells_mask]))
        
        # 计算最终相似度
        similarities = 1 - (1 / (2 * d)) * ecs
        return torch.mean(similarities)
    
    def _similarity_matrix_batch_cpu(self, patterns_batch: List[np.ndarray], d: float = 0.9):
        """CPU批量相似度计算的回退方案"""
        n_patterns = len(patterns_batch)
        similarity_matrix = np.eye(n_patterns, dtype=np.float32)
        
        for i in range(n_patterns):
            for j in range(i+1, n_patterns):
                sim = self._similarity_matrix_v2(patterns_batch[i], patterns_batch[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        return similarity_matrix 