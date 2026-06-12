# This is the fastest python implementation of the ForceAtlas2 plugin from Gephi
# intended to be used with networkx, but is in theory independent of
# it since it only relies on the adjacency matrix.  This
# implementation is based directly on the Gephi plugin:
#
# https://github.com/gephi/gephi/blob/master/modules/LayoutPlugin/src/main/java/org/gephi/layout/plugin/forceAtlas2/ForceAtlas2.java
#
# For simplicity and for keeping code in sync with upstream, I have
# reused as many of the variable/function names as possible, even when
# they are in a more java-like style (e.g. camelcase)
#
# I wrote this because I wanted an almost feature complete and fast implementation
# of ForceAtlas2 algorithm in python
#
# NOTES: Currently, this only works for weighted undirected graphs.
#
# Copyright (C) 2017 Bhargav Chippada <bhargavchippada19@gmail.com>
#
# Available under the GPLv3

import random
import time
import math

import numpy
import scipy
from tqdm import tqdm
import concurrent.futures  # Added for multithreading support
import warnings

from ..._optional import normalize_torch_device
from . import fa2util


class Timer:
    def __init__(self, name="Timer"):
        self.name = name
        self.start_time = 0.0
        self.total_time = 0.0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.total_time += (time.time() - self.start_time)

    def display(self):
        print(self.name, " took ", "%.2f" % self.total_time, " seconds")


class ForceAtlas2:
    def __init__(self,
                 # Behavior alternatives
                 outboundAttractionDistribution=False,  # Dissuade hubs
                 linLogMode=False,  # NOT IMPLEMENTED
                 adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                 edgeWeightInfluence=1.0,

                 # Performance
                 jitterTolerance=1.0,  # Tolerance
                 barnesHutOptimize=True,
                 barnesHutTheta=1.2,
                 multiThreaded=False,  # Use multithreading for repulsion forces
                 numThreads=None,  # Number of threads to use (None = auto)
                 useGPU=False,     # Use GPU acceleration if available
                 optimizedThreading=True,  # Use optimized threading algorithm
                 
                 # Tuning
                 scalingRatio=2.0,
                 strongGravityMode=False,
                 gravity=1.0,

                 # Log
                 verbose=True):
        assert linLogMode == adjustSizes == False, "You selected a feature that has not been implemented yet..."
        self.outboundAttractionDistribution = outboundAttractionDistribution
        self.linLogMode = linLogMode
        self.adjustSizes = adjustSizes
        self.edgeWeightInfluence = edgeWeightInfluence
        self.jitterTolerance = jitterTolerance
        self.barnesHutOptimize = barnesHutOptimize
        self.barnesHutTheta = barnesHutTheta
        self.multiThreaded = multiThreaded
        self.numThreads = numThreads
        self.useGPU = useGPU
        self.optimizedThreading = optimizedThreading
        self.scalingRatio = scalingRatio
        self.strongGravityMode = strongGravityMode
        self.gravity = gravity
        self.verbose = verbose
        
        # Initialize PyTorch if requested
        self.torch_available = False
        self.device = None
        self.using_cython = False
        
        # Try to detect Cython implementation
        try:
            import fa2util
            if hasattr(fa2util, '__file__') and fa2util.__file__.endswith(('.so', '.pyd', '.dylib')):
                self.using_cython = True
                if self.verbose:
                    print("✅ Using compiled Cython implementation for maximum performance")
            else:
                if self.verbose:
                    print("⚠️ Using pure Python implementation. Compile with Cython for 10-100x speedup.")
                    print("   Run the compile_fa2.py script or install with: pip install -e .")
        except ImportError:
            if self.verbose:
                print("⚠️ Using pure Python implementation. Compile with Cython for 10-100x speedup.")
        
        # Check GPU availability if requested
        if self.useGPU:
            try:
                import torch
                self.torch = torch  # Store for later use
                
                # More robust GPU detection
                cuda_available = torch.cuda.is_available()
                mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                
                if cuda_available:
                    if self.verbose:
                        print(f"✅ Using CUDA GPU acceleration: {torch.cuda.get_device_name(0)}")
                    self.device = normalize_torch_device("cuda")
                    self.torch_available = True
                elif mps_available:
                    if self.verbose:
                        print("✅ Using Apple Silicon MPS acceleration")
                    self.device = torch.device("mps")
                    self.torch_available = True
                else:
                    if self.verbose:
                        print("⚠️ GPU acceleration requested but no GPU found. Falling back to CPU.")
                    self.device = torch.device("cpu")
                    # No need to set torch_available to True here as we're using CPU
                
            except ImportError:
                if self.verbose:
                    print("⚠️ GPU acceleration requested but PyTorch not installed.")
                    print("   Install with: pip install torch")
                self.torch_available = False
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error initializing GPU: {str(e)}. Falling back to CPU.")
                self.torch_available = False
        
        # Diagnostics
        if self.verbose:
            if self.using_cython and self.torch_available:
                print("🚀 Using both Cython and GPU acceleration for maximum performance")
            elif self.using_cython:
                print("🚀 Using Cython acceleration")
            elif self.torch_available:
                print("🚀 Using GPU acceleration")
                
            if self.multiThreaded:
                import multiprocessing
                max_threads = multiprocessing.cpu_count()
                actual_threads = self.numThreads if self.numThreads else max_threads
                print(f"🧵 Using multithreading with {actual_threads}/{max_threads} threads")

    def init(self,
             G,  # a graph in 2D numpy ndarray format (or) scipy sparse matrix format
             pos=None  # Array of initial positions
             ):
        isSparse = False
        if isinstance(G, numpy.ndarray):
            # Check our assumptions
            assert G.shape == (G.shape[0], G.shape[0]), "G is not 2D square"
            assert numpy.all(G.T == G), "G is not symmetric.  Currently only undirected graphs are supported"
            assert isinstance(pos, numpy.ndarray) or (pos is None), "Invalid node positions"
        elif scipy.sparse.issparse(G):
            # Check our assumptions
            assert G.shape == (G.shape[0], G.shape[0]), "G is not 2D square"
            assert isinstance(pos, numpy.ndarray) or (pos is None), "Invalid node positions"
            G = G.tolil()
            isSparse = True
        else:
            assert False, "G is not numpy ndarray or scipy sparse matrix"

        # Put nodes into a data structure we can understand
        nodes = []
        for i in range(0, G.shape[0]):
            n = fa2util.Node()
            if isSparse:
                n.mass = 1 + len(G.rows[i])
            else:
                n.mass = 1 + numpy.count_nonzero(G[i])
            n.old_dx = 0
            n.old_dy = 0
            n.dx = 0
            n.dy = 0
            if pos is None:
                n.x = random.random()
                n.y = random.random()
            else:
                n.x = pos[i][0]
                n.y = pos[i][1]
            nodes.append(n)

        # Put edges into a data structure we can understand
        edges = []
        es = numpy.asarray(G.nonzero()).T
        for e in es:  # Iterate through edges
            if e[1] <= e[0]: continue  # Avoid duplicate edges
            edge = fa2util.Edge()
            edge.node1 = e[0]  # The index of the first node in `nodes`
            edge.node2 = e[1]  # The index of the second node in `nodes`
            edge.weight = G[tuple(e)]
            edges.append(edge)

        return nodes, edges
        
    def _init_gpu_structures(self, nodes, edges):
        """Initialize PyTorch tensors for GPU computation"""
        import torch
        
        n_nodes = len(nodes)
        
        # Initialize node positions and properties
        positions = torch.zeros((n_nodes, 2), device=self.device, dtype=torch.float32)
        old_forces = torch.zeros((n_nodes, 2), device=self.device, dtype=torch.float32)
        forces = torch.zeros((n_nodes, 2), device=self.device, dtype=torch.float32)
        masses = torch.zeros(n_nodes, device=self.device, dtype=torch.float32)
        
        # Fill in values from nodes
        for i, n in enumerate(nodes):
            positions[i, 0] = n.x
            positions[i, 1] = n.y
            old_forces[i, 0] = n.old_dx
            old_forces[i, 1] = n.old_dy
            masses[i] = n.mass
        
        # Initialize edge structure
        if edges:
            n_edges = len(edges)
            edge_sources = torch.zeros(n_edges, device=self.device, dtype=torch.long)
            edge_targets = torch.zeros(n_edges, device=self.device, dtype=torch.long)
            edge_weights = torch.zeros(n_edges, device=self.device, dtype=torch.float32)
            
            for i, e in enumerate(edges):
                edge_sources[i] = e.node1
                edge_targets[i] = e.node2
                edge_weights[i] = e.weight
                
            return positions, old_forces, forces, masses, edge_sources, edge_targets, edge_weights
        else:
            return positions, old_forces, forces, masses, None, None, None

    def _apply_repulsion_gpu(self, positions, forces, masses, coefficient):
        """Apply repulsion forces between all nodes using GPU acceleration"""
        import torch
        
        n_nodes = positions.shape[0]
        
        # Compute all pairwise distances
        # Create expanded tensors for vectorized computation
        pos_i = positions.unsqueeze(1)  # [n_nodes, 1, 2]
        pos_j = positions.unsqueeze(0)  # [1, n_nodes, 2]
        
        # Compute squared distance between all pairs
        diff = pos_i - pos_j  # [n_nodes, n_nodes, 2]
        dist_squared = torch.sum(diff**2, dim=2)  # [n_nodes, n_nodes]
        
        # Create mask to avoid self-repulsion
        mask = torch.ones_like(dist_squared, dtype=torch.bool)
        mask.fill_diagonal_(0)
        
        # Avoid division by zero
        dist_squared = torch.where(dist_squared > 0, dist_squared, torch.ones_like(dist_squared))
        
        # Compute force factors
        mass_i = masses.unsqueeze(1)  # [n_nodes, 1]
        mass_j = masses.unsqueeze(0)  # [1, n_nodes]
        factor = coefficient * mass_i * mass_j / dist_squared
        
        # Zero out self-interactions
        factor = factor * mask
        
        # Apply forces
        force_x = diff[:, :, 0] * factor  # [n_nodes, n_nodes]
        force_y = diff[:, :, 1] * factor  # [n_nodes, n_nodes]
        
        # Sum forces for each node
        forces[:, 0] += torch.sum(force_x, dim=1)
        forces[:, 1] += torch.sum(force_y, dim=1)
        
        return forces

    def _apply_gravity_gpu(self, positions, forces, masses, gravity, use_strong_gravity=False):
        """Apply gravity forces using GPU acceleration"""
        import torch
        
        # Calculate distance from center
        dist = torch.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        
        # Avoid division by zero
        dist = torch.where(dist > 0, dist, torch.ones_like(dist))
        
        if use_strong_gravity:
            # Strong gravity - distance doesn't matter
            factor = self.scalingRatio * masses * gravity
        else:
            # Regular gravity - force decreases with distance
            factor = masses * gravity / dist
        
        # Apply force toward center (0,0)
        forces[:, 0] -= positions[:, 0] * factor
        forces[:, 1] -= positions[:, 1] * factor
        
        return forces

    def _apply_attraction_gpu(self, positions, forces, masses, edge_sources, edge_targets, edge_weights, coefficient, distributed_attraction):
        """Apply attraction forces between connected nodes using GPU acceleration"""
        import torch
        
        # Get positions of source and target nodes
        source_pos = positions[edge_sources]  # [n_edges, 2]
        target_pos = positions[edge_targets]  # [n_edges, 2]
        
        # Compute distance vector
        diff = source_pos - target_pos  # [n_edges, 2]
        
        if self.edgeWeightInfluence != 1.0:
            edge_weights = edge_weights ** self.edgeWeightInfluence
            
        if distributed_attraction:
            # Distributed attraction - hubs attract less
            source_masses = masses[edge_sources]  # [n_edges]
            factor = -coefficient * edge_weights.unsqueeze(1) / source_masses.unsqueeze(1)  # [n_edges, 1]
        else:
            # Regular attraction
            factor = -coefficient * edge_weights.unsqueeze(1)  # [n_edges, 1]
        
        # Compute forces
        edge_forces = diff * factor  # [n_edges, 2]
        
        # Sum forces for each node using scatter_add
        for i in range(edge_sources.size(0)):
            forces[edge_sources[i]] += edge_forces[i]
            forces[edge_targets[i]] -= edge_forces[i]
        
        return forces

    def _adjust_speeds_gpu(self, positions, forces, old_forces, masses, speed, speed_efficiency, jitter_tolerance):
        """Adjust speeds and apply forces on the GPU"""
        import torch
        
        # Calculate swinging and effective traction
        swinging = torch.sqrt(((old_forces - forces) ** 2).sum(dim=1))
        effective_traction = 0.5 * torch.sqrt(((old_forces + forces) ** 2).sum(dim=1))
        
        # Calculate total swinging and effective traction
        total_swinging = (masses * swinging).sum()
        total_effective_traction = (masses * effective_traction).sum()
        
        # Optimize jitter tolerance
        # The 'right' jitter tolerance for this network is totalSwinging / totalEffectiveTraction
        # But given the current jitterTolerance, compute the adaptation
        estimated_optimal_jitter_tolerance = 0.05 * total_swinging / (total_effective_traction + 1e-10)
        
        # Limit to reasonable bounds
        min_jt = torch.tensor(0.05, device=self.device)
        max_jt = torch.tensor(2.0, device=self.device)
        jt = jitter_tolerance * torch.max(min_jt, torch.min(max_jt, estimated_optimal_jitter_tolerance))
        
        min_speed_efficiency = torch.tensor(0.05, device=self.device)
        
        # Protection against erratic behavior
        if total_swinging / (total_effective_traction + 1e-10) > 2.0:
            # If swinging is big, it's erratic
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency = speed_efficiency * 0.5
            jt = torch.max(jt, torch.tensor(jitter_tolerance, device=self.device))
        
        # Adjust speed
        target_speed = jt * speed_efficiency * total_effective_traction / (total_swinging + 1e-10)
        
        # Speed shouldn't rise too much too quickly
        max_rise = torch.tensor(0.5, device=self.device)
        speed = speed + torch.min(target_speed - speed, max_rise * speed)
        
        # Apply forces to update positions
        # Calculate the factor for each node
        swinging_factor = torch.sqrt((old_forces - forces).pow(2).sum(dim=1))
        factor = speed / (1.0 + torch.sqrt(speed * swinging_factor))
        
        # Unsqueeze for broadcasting
        factor = factor.unsqueeze(1)
        
        # Update positions
        positions = positions + forces * factor
        
        # Return the updated speed values as Python floats
        return positions, speed.item(), speed_efficiency.item()

    def _sync_gpu_to_nodes(self, nodes, positions, forces):
        """Sync GPU tensor data back to node objects"""
        import torch
        
        positions_cpu = positions.cpu().numpy()
        forces_cpu = forces.cpu().numpy()
        
        for i, n in enumerate(nodes):
            n.x = positions_cpu[i, 0]
            n.y = positions_cpu[i, 1]
            n.dx = forces_cpu[i, 0]
            n.dy = forces_cpu[i, 1]
            
    def _sync_nodes_to_gpu(self, nodes, positions, forces, old_forces):
        """Sync node data to GPU tensors"""
        import torch
        
        for i, n in enumerate(nodes):
            positions[i, 0] = n.x
            positions[i, 1] = n.y
            forces[i, 0] = n.dx
            forces[i, 1] = n.dy
            old_forces[i, 0] = n.old_dx
            old_forces[i, 1] = n.old_dy

    # Given an adjacency matrix, this function computes the node positions
    # according to the ForceAtlas2 layout algorithm.  It takes the same
    # arguments that one would give to the ForceAtlas2 algorithm in Gephi.
    # Not all of them are implemented.  See below for a description of
    # each parameter and whether or not it has been implemented.
    #
    # This function will return a list of X-Y coordinate tuples, ordered
    # in the same way as the rows/columns in the input matrix.
    #
    # The only reason you would want to run this directly is if you don't
    # use networkx.  In this case, you'll likely need to convert the
    # output to a more usable format.  If you do use networkx, use the
    # "forceatlas2_networkx_layout" function below.
    #
    # Currently, only undirected graphs are supported so the adjacency matrix
    # should be symmetric.
    def forceatlas2(self,
                    G,  # a graph in 2D numpy ndarray format (or) scipy sparse matrix format
                    pos=None,  # Array of initial positions
                    iterations=100  # Number of times to iterate the main loop
                    ):
        # Initializing, initAlgo()
        # ================================================================

        # speed and speedEfficiency describe a scaling factor of dx and dy
        # before x and y are adjusted.  These are modified as the
        # algorithm runs to help ensure convergence.
        speed = 1.0
        speedEfficiency = 1.0
        nodes, edges = self.init(G, pos)
        
        # Use GPU if available and requested
        using_gpu = self.torch_available and self.useGPU
        gpu_structures = None
        
        if using_gpu:
            try:
                gpu_structures = self._init_gpu_structures(nodes, edges)
                if self.verbose:
                    print("✅ GPU structures initialized successfully")
            except Exception as e:
                using_gpu = False
                if self.verbose:
                    print(f"⚠️ Failed to initialize GPU structures: {str(e)}")
                    print("   Falling back to CPU implementation")

        # If Barnes Hut active, initialize root region
        if self.barnesHutOptimize:
            rootRegion = fa2util.Region(nodes)
            if hasattr(rootRegion, 'buildSubRegions'):
                rootRegion.buildSubRegions()
            else:
                if self.verbose:
                    print("⚠️ Barnes-Hut optimization requires compiled Cython implementation")
                    print("   Run the compile_fa2.py script to enable it")
                self.barnesHutOptimize = False

        # Prepare for multithreaded execution if requested
        using_multithreading = self.multiThreaded
        thread_pool = None
        
        if using_multithreading:
            try:
                import concurrent.futures
                import multiprocessing
                
                # Determine number of threads
                max_threads = multiprocessing.cpu_count()
                if self.numThreads is None:
                    self.numThreads = max_threads
                else:
                    self.numThreads = min(self.numThreads, max_threads)
                
                # Create thread pool
                thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.numThreads)
                
                # Prepare work distribution
                if self.optimizedThreading:
                    work_chunks = self._split_work_for_threading(len(nodes))
                else:
                    work_chunks = self._split_nodes_for_threading(nodes)
                    
            except Exception as e:
                using_multithreading = False
                if self.verbose:
                    print(f"⚠️ Failed to initialize multithreading: {str(e)}")
                    print("   Falling back to single-threaded implementation")

        # Main loop
        # =====================
        if self.verbose:
            from tqdm import tqdm
            progress_bar = tqdm(range(0, iterations), desc="ForceAtlas2")
        else:
            progress_bar = range(0, iterations)
            
        for _i in progress_bar:
            # Calculate repulsion using Barnes Hut optimization if enabled
            if self.barnesHutOptimize:
                # Reset forces
                for n in nodes:
                    n.dx = 0
                    n.dy = 0
                
                # Apply forces using Barnes Hut
                if using_multithreading:
                    futures = []
                    for start_idx, end_idx in work_chunks:
                        futures.append(
                            thread_pool.submit(
                                self._apply_barnes_hut_on_nodes,
                                rootRegion,
                                nodes[start_idx:end_idx],
                                len(futures)
                            )
                        )
                    # Wait for all threads to complete
                    concurrent.futures.wait(futures)
                else:
                    # Single-threaded version
                    rootRegion.applyForceOnNodes(nodes, self.barnesHutTheta, self.scalingRatio)
            else:
                # N-body repulsion (without Barnes Hut)
                if using_gpu:
                    # Use GPU for repulsion
                    try:
                        self._apply_repulsion_gpu(*gpu_structures, self.scalingRatio)
                    except Exception as e:
                        if self.verbose:
                            print(f"⚠️ GPU repulsion calculation failed: {str(e)}")
                            print("   Falling back to CPU for this iteration")
                        
                        # Reset forces
                        for n in nodes:
                            n.dx = 0
                            n.dy = 0
                            
                        # Fall back to CPU implementation just for this iteration
                        if using_multithreading:
                            futures = []
                            for i, (start_idx, end_idx) in enumerate(work_chunks):
                                futures.append(
                                    thread_pool.submit(
                                        self._compute_repulsion_chunk,
                                        nodes[start_idx:end_idx],
                                        nodes,
                                        start_idx,
                                        end_idx,
                                        self.scalingRatio,
                                        i
                                    )
                                )
                            concurrent.futures.wait(futures)
                        else:
                            fa2util.apply_repulsion(nodes, self.scalingRatio)
                elif using_multithreading:
                    # Use CPU multithreading for repulsion
                    futures = []
                    for i, (start_idx, end_idx) in enumerate(work_chunks):
                        futures.append(
                            thread_pool.submit(
                                self._compute_repulsion_chunk,
                                nodes[start_idx:end_idx],
                                nodes,
                                start_idx,
                                end_idx,
                                self.scalingRatio,
                                i
                            )
                        )
                    concurrent.futures.wait(futures)
                else:
                    # Use single-threaded CPU implementation
                    fa2util.apply_repulsion(nodes, self.scalingRatio)

            # Apply gravity
            if using_gpu:
                try:
                    self._apply_gravity_gpu(*gpu_structures[:4], self.gravity, self.strongGravityMode)
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ GPU gravity calculation failed: {str(e)}")
                    # Fall back to CPU
                    fa2util.apply_gravity(nodes, self.gravity, self.scalingRatio, self.strongGravityMode)
            else:
                fa2util.apply_gravity(nodes, self.gravity, self.scalingRatio, self.strongGravityMode)

            # Apply attraction
            if using_gpu and gpu_structures[4] is not None:  # Check if we have edge data
                try:
                    self._apply_attraction_gpu(*gpu_structures, self.scalingRatio, self.outboundAttractionDistribution)
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ GPU attraction calculation failed: {str(e)}")
                    # Fall back to CPU
                    fa2util.apply_attraction(nodes, edges, self.outboundAttractionDistribution, 
                                           self.scalingRatio, self.edgeWeightInfluence)
            else:
                fa2util.apply_attraction(nodes, edges, self.outboundAttractionDistribution, 
                                       self.scalingRatio, self.edgeWeightInfluence)

            # Auto adjust speed
            if using_gpu:
                try:
                    speed, speedEfficiency = self._adjust_speeds_gpu(*gpu_structures[:4], speed, speedEfficiency, self.jitterTolerance)
                    # Sync GPU data back to nodes
                    self._sync_gpu_to_nodes(nodes, *gpu_structures[:2])
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ GPU speed adjustment failed: {str(e)}")
                    # Fall back to CPU
                    values = fa2util.adjustSpeedAndApplyForces(nodes, speed, speedEfficiency, self.jitterTolerance)
                    speed = values['speed']
                    speedEfficiency = values['speedEfficiency']
            else:
                values = fa2util.adjustSpeedAndApplyForces(nodes, speed, speedEfficiency, self.jitterTolerance)
                speed = values['speed']
                speedEfficiency = values['speedEfficiency']
                
            # Update root region if using Barnes Hut
            if self.barnesHutOptimize:
                rootRegion = fa2util.Region(nodes)
                rootRegion.buildSubRegions()

            # Sync back to GPU if needed
            if using_gpu:
                try:
                    self._sync_nodes_to_gpu(nodes, *gpu_structures[:3])
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Failed to sync nodes to GPU: {str(e)}")
                    # This is a more serious error, disable GPU for next iterations
                    using_gpu = False
        
        # Clean up
        if thread_pool:
            thread_pool.shutdown()

        return [(n.x, n.y) for n in nodes]

    def _split_work_for_threading(self, n_nodes):
        """Split node pair calculations into balanced chunks for threading
        
        Returns a list of (start_idx, end_idx) tuples for each thread
        """
        import multiprocessing
        
        # Number of node pairs to process (upper triangular matrix without diagonal)
        total_pairs = (n_nodes * (n_nodes - 1)) // 2
        
        if self.numThreads is None:
            # Use the number of available CPU cores if not specified
            num_threads = multiprocessing.cpu_count()
        else:
            num_threads = self.numThreads
        
        # Ensure we don't create more threads than work items
        num_threads = min(num_threads, total_pairs)
        
        # Calculate chunks
        chunk_size = total_pairs // num_threads
        remainder = total_pairs % num_threads
        
        chunks = []
        start = 0
        
        for i in range(num_threads):
            # Add one extra item to the first 'remainder' chunks
            size = chunk_size + (1 if i < remainder else 0)
            end = start + size
            chunks.append((start, end))
            start = end
            
        return chunks

    def _apply_barnes_hut_on_nodes(self, region, nodes_chunk, thread_id=0):
        """Apply Barnes Hut force calculation on a subset of nodes"""
        if self.verbose:
            for n in nodes_chunk:
                region.applyForce(n, self.barnesHutTheta, self.scalingRatio)
        else:
            region.applyForceOnNodes(nodes_chunk, self.barnesHutTheta, self.scalingRatio)
            
    def _compute_repulsion_chunk(self, nodes, all_nodes, start_idx, end_idx, coefficient, thread_id=0):
        """Compute repulsion for a specific chunk of node pairs
        
        Instead of dividing nodes, this divides the work more efficiently by index ranges.
        This method handles n*(n-1)/2 pairs of interaction without duplicates.
        """
        n = len(nodes)
        
        if self.verbose:
            iterator = tqdm(range(start_idx, end_idx), desc=f"Repulsion Thread {thread_id}", leave=False, position=thread_id+1)
        else:
            iterator = range(start_idx, end_idx)
        
        # 预计算一些常量，避免循环中重复计算
        k = int((-1 + math.sqrt(1 + 8 * start_idx)) / 2)
        i_offset = start_idx - k * (k + 1) // 2
        
        # 使用更高效的算法直接计算行列
        for idx in iterator:
            i_rel = idx - start_idx
            
            # 增量计算行列，避免每次都使用复杂的二次方程公式
            i = k + (i_rel + i_offset) // (n - k - 1)
            j = (i_rel + i_offset) % (n - k - 1) + k + 1
            
            # 确保 i < j 且在范围内
            if i < j and i < n and j < n:
                # 应用节点之间的斥力
                fa2util.linRepulsion(nodes[i], nodes[j], coefficient)

    def _split_nodes_for_threading(self, nodes):
        """Split the nodes into chunks for parallel processing"""
        import multiprocessing
        
        if self.numThreads is None:
            # Use the number of available CPU cores if not specified
            num_threads = multiprocessing.cpu_count()
        else:
            num_threads = self.numThreads
            
        # Calculate chunk size (at least 1 node per chunk)
        chunk_size = max(1, len(nodes) // num_threads)
        
        # Split nodes into chunks
        node_chunks = []
        for i in range(0, len(nodes), chunk_size):
            chunk = nodes[i:i + chunk_size]
            if chunk:  # Ensure we don't add empty chunks
                node_chunks.append(chunk)
                
        return node_chunks

    # A layout for NetworkX.
    #
    # This function returns a NetworkX layout, which is really just a
    # dictionary of node positions (2D X-Y tuples) indexed by the node name.
    def forceatlas2_networkx_layout(self, G, pos=None, iterations=100, weight_attr=None):
        import networkx
        try:
            import cynetworkx
        except ImportError:
            cynetworkx = None

        assert (
            isinstance(G, networkx.classes.graph.Graph)
            or (cynetworkx and isinstance(G, cynetworkx.classes.graph.Graph))
        ), "Not a networkx graph"
        assert isinstance(pos, dict) or (pos is None), "pos must be specified as a dictionary, as in networkx"
        M = networkx.to_scipy_sparse_array(G, dtype='f', format='lil', weight=weight_attr)
        if pos is None:
            l = self.forceatlas2(M, pos=None, iterations=iterations)
        else:
            poslist = numpy.asarray([pos[i] for i in G.nodes()])
            l = self.forceatlas2(M, pos=poslist, iterations=iterations)
        return dict(zip(G.nodes(), l))

    # A layout for igraph.
    #
    # This function returns an igraph layout
    def forceatlas2_igraph_layout(self, G, pos=None, iterations=100, weight_attr=None):

        from scipy.sparse import csr_matrix
        import igraph

        def to_sparse(graph, weight_attr=None):
            edges = graph.get_edgelist()
            if weight_attr is None:
                weights = [1] * len(edges)
            else:
                weights = graph.es[weight_attr]

            if not graph.is_directed():
                edges.extend([(v, u) for u, v in edges])
                weights.extend(weights)

            return csr_matrix((weights, list(zip(*edges))))

        assert isinstance(G, igraph.Graph), "Not a igraph graph"
        assert isinstance(pos, (list, numpy.ndarray)) or (pos is None), "pos must be a list or numpy array"

        if isinstance(pos, list):
            pos = numpy.array(pos)

        adj = to_sparse(G, weight_attr)
        coords = self.forceatlas2(adj, pos=pos, iterations=iterations)

        return igraph.layout.Layout(coords, 2)
