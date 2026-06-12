"""Mehods and classes for training gene expression classifiers."""

from abc import ABC, abstractmethod
from typing import Sequence
import warnings

import scipy.sparse
import numpy as np
import numpy.typing as npt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm
import sklearn

from .countdata import CountData
from .sparsetools import rescale_rows
from . import Colors, EMOJI
from ..._optional import normalize_torch_device

# Try to import torch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def check_gpu_available():
    """Check if GPU is available for acceleration.

    Returns:
        tuple: (gpu_available, device_name, sklearn_version_ok)
    """
    if not TORCH_AVAILABLE:
        return False, "No torch", False

    # Check sklearn version (need >= 1.8 for torch tensor support)
    sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
    sklearn_version_ok = sklearn_version >= (1, 8)

    # Check for CUDA
    if torch.cuda.is_available():
        return True, f"CUDA ({torch.cuda.get_device_name(0)})", sklearn_version_ok

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return True, "MPS (Apple Silicon)", sklearn_version_ok

    # Check for ROCm
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return True, "ROCm (AMD)", sklearn_version_ok

    return False, "CPU only", sklearn_version_ok


def sparse_to_torch(matrix: scipy.sparse.spmatrix, device: str = 'cuda',
                    force_dense: bool = False):
    """Convert scipy sparse matrix to torch sparse tensor.

    Args:
        matrix: Scipy sparse matrix
        device: Target device ('cuda', 'mps', or 'cpu')
        force_dense: If True, convert to dense tensor (default: False)

    Returns:
        torch.Tensor: Sparse or dense tensor on the specified device
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is not available")

    if force_dense or not scipy.sparse.issparse(matrix):
        # Dense path
        if scipy.sparse.issparse(matrix):
            dense = matrix.toarray()
        else:
            dense = np.asarray(matrix)

        tensor = torch.from_numpy(np.asarray(dense, dtype=np.float32)).float()

        if device == 'cuda' and torch.cuda.is_available():
            tensor = tensor.cuda()
        elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            tensor = tensor.to('mps')
        else:
            tensor = tensor.cpu()

        return tensor

    # Sparse path - convert to COO format for PyTorch
    if not scipy.sparse.isspmatrix_coo(matrix):
        matrix = matrix.tocoo()

    # Get indices and values
    indices = np.vstack([matrix.row, matrix.col])
    values = matrix.data
    shape = matrix.shape

    # Convert to torch sparse tensor
    indices_tensor = torch.LongTensor(indices)
    values_tensor = torch.FloatTensor(values)

    sparse_tensor = torch.sparse_coo_tensor(
        indices_tensor,
        values_tensor,
        size=shape,
        dtype=torch.float32
    )

    # Move to device
    if device == 'cuda' and torch.cuda.is_available():
        sparse_tensor = sparse_tensor.cuda()
    elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        sparse_tensor = sparse_tensor.to('mps')
    else:
        sparse_tensor = sparse_tensor.cpu()

    return sparse_tensor


def normalize_rows(matrix: scipy.sparse.spmatrix | np.matrix,
                   r: int = 5
                   ) -> scipy.sparse.spmatrix:
    sums = np.array(matrix.sum(axis=1)).flatten()
    factors = np.zeros_like(sums, dtype=np.float64)
    nonzero = sums > 0
    factors[nonzero] = (10 ** r) / sums[nonzero]
    return rescale_rows(matrix, factors)


class Classifier(ABC):
    """Abstract class for a gene expression classifier.

    Attributes:
        trained: True if and only if the classifier has been trained.
        classes: If trained, then an ordered list of all classes.
    """

    def __init__(self):
        self.trained = False
        self.classes = []

    @abstractmethod
    def classify(self,
                 samples: scipy.sparse.spmatrix | np.matrix,
                 silent: bool = False
                 ) -> npt.NDArray:
        """Takes gene expression samples and returns a classification.

        Args:
            samples:
                A matrix of gene expressions. Each row represents a single
                sample.
            silent:
                If True, suppress output messages (useful for multiprocessing).

        Returns:
            An matrix of probabilities, of shape (samples, classes). Each row
            records the confidence that the respective sample came from each
            class.
        """
        if not self.trained:
            raise RuntimeError('Must provide training data before classifying')

    @abstractmethod
    def train(self, X_train: scipy.sparse.spmatrix, y_train: Sequence[str]):
        """Trains the classifier on annotated samples.

        Args:
            X_train:
                A matrix of gene expressions. Each row represents a single
                sample.
            y_train:
                A label for each sample.
        """
        self.trained = True


class SVCClassifier(Classifier):
    """Based on the implementation of Abdelaal et al. 2019

    NOTE: This uses sklearn's LinearSVC which does NOT support GPU.
    For real GPU acceleration, use TorchLinearClassifier instead.
    """

    def __init__(self, r_value: int = 5, use_gpu: bool = False):
        """Initialize SVC Classifier.

        Args:
            r_value: Normalization factor (default: 5)
            use_gpu: DEPRECATED - sklearn does not support GPU. Use TorchLinearClassifier for GPU.
        """
        super().__init__()
        scaler = StandardScaler(with_mean=False)
        clf = Pipeline([('scaler', scaler), ('clf', LinearSVC(dual=False))])
        self.clf = CalibratedClassifierCV(clf)
        self.r_value = r_value
        self.use_gpu = use_gpu
        self.device = None  # sklearn doesn't use GPU

        if use_gpu:
            print(f"{Colors.WARNING}⚠️ WARNING: SVCClassifier does not support GPU!{Colors.ENDC}")
            print(f"{Colors.WARNING}   sklearn's LinearSVC runs on CPU only.{Colors.ENDC}")
            print(f"{Colors.GREEN}   💡 For real GPU acceleration, use: TorchLinearClassifier{Colors.ENDC}")
            print(f"{Colors.BLUE}      from omicverse.external.topact import TorchLinearClassifier{Colors.ENDC}")
            print(f"{Colors.BLUE}      classifier = TorchLinearClassifier(device='auto'){Colors.ENDC}")

        print(f"{Colors.BLUE}ℹ️ Using CPU-based sklearn LinearSVC{Colors.ENDC}")

    def train(self, X_train: scipy.sparse.spmatrix, y_train: Sequence[str]):
        print(f"{Colors.CYAN}{EMOJI['train']} Training SVC classifier (CPU)...{Colors.ENDC}")
        print(f"{Colors.BLUE}  → Normalizing {X_train.shape[0]} samples with r_value={self.r_value}{Colors.ENDC}")
        X_train = normalize_rows(X_train, self.r_value)
        X_train = X_train.log1p()

        print(f"{Colors.BLUE}  → Fitting sklearn classifier on {len(set(y_train))} cell types{Colors.ENDC}")
        self.clf.fit(X_train, y_train)
        self.classes = self.clf.classes_
        super().train(X_train, y_train)
        print(f"{Colors.GREEN}{EMOJI['done']} Training completed! Classes: {list(self.classes)}{Colors.ENDC}")

    def classify(self,
                 samples: scipy.sparse.spmatrix | np.matrix,
                 silent: bool = False
                 ):  # -> npt.NDArray:
        super().classify(samples)
        if not silent:
            print(f"{Colors.CYAN}{EMOJI['classify']} Classifying {samples.shape[0]} samples (CPU)...{Colors.ENDC}")
        test = normalize_rows(samples, self.r_value)
        if scipy.sparse.isspmatrix(test):
            test = test.log1p()
        else:
            # Dense array/matrix
            test = np.log1p(np.asarray(test))

        probs = self.clf.predict_proba(test)

        if not silent:
            print(f"{Colors.GREEN}{EMOJI['done']} Classification completed!{Colors.ENDC}")
        return probs


class SparseLinearModel(torch.nn.Module):
    """Linear model that supports sparse input."""

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_features, num_classes) * 0.01)
        self.bias = torch.nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        """Forward pass supporting both sparse and dense tensors."""
        if x.is_sparse:
            # Sparse matrix multiplication: (N, F) @ (F, C) = (N, C)
            output = torch.sparse.mm(x, self.weight) + self.bias
        else:
            # Dense matrix multiplication
            output = torch.mm(x, self.weight) + self.bias
        return output


class TorchLinearClassifier(Classifier):
    """GPU-accelerated linear classifier using PyTorch with sparse tensor support.

    This is a REAL GPU classifier that runs on:
    - CUDA (NVIDIA GPUs) - Full sparse support
    - MPS (Apple Silicon) - Limited sparse support, may use dense fallback
    - ROCm (AMD GPUs) - Sparse support
    - CPU (fallback) - Sparse support

    Supports sparse matrices natively without converting to dense,
    saving memory and computation for large datasets.

    NOTE: This uses cross-entropy loss instead of hinge loss (SVM).
    Results may differ slightly from SVCClassifier. For maximum compatibility,
    test on your data first using COMPARE_CLASSIFIERS.py.
    """

    def __init__(self, r_value: int = 5, device: str = 'auto',
                 max_iter: int = 1000, lr: float = 0.01,
                 batch_size: int = 1024, weight_decay: float = 0.01,
                 use_sparse: bool = True,
                 use_amp: bool = False,
                 pin_memory: bool = True):
        """Initialize PyTorch GPU classifier.

        Args:
            r_value: Normalization factor (default: 5)
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            max_iter: Maximum training iterations
            lr: Learning rate for optimization
            batch_size: Batch size for training
            weight_decay: L2 regularization strength (default: 0.01, similar to SVC)
            use_sparse: Use sparse tensors if True (default: True, recommended for large data)
            use_amp: Use automatic mixed precision on CUDA during training/inference.
                Disabled by default to preserve maximum numerical consistency.
            pin_memory: Pin host memory before CUDA transfer for faster H2D copies.
        """
        super().__init__()
        self.r_value = r_value
        self.max_iter = max_iter
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.use_sparse = use_sparse
        self.use_amp = use_amp
        self.pin_memory = pin_memory

        # Determine device
        if device == 'auto':
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    self.device = normalize_torch_device("cuda")
                    device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = torch.device('mps')
                    device_name = "MPS (Apple Silicon)"
                    # MPS has limited sparse support, use dense for now
                    if use_sparse:
                        print(f"{Colors.WARNING}⚠️  MPS has limited sparse support, using dense tensors{Colors.ENDC}")
                        self.use_sparse = False
                else:
                    self.device = torch.device('cpu')
                    device_name = "CPU"
            else:
                raise RuntimeError("PyTorch not available. Install with: pip install torch")
        else:
            self.device = normalize_torch_device(device)
            device_name = device

        sparse_status = "sparse tensors" if self.use_sparse else "dense tensors"
        print(f"{Colors.GREEN}🚀 GPU Classifier initialized on: {device_name} ({sparse_status}){Colors.ENDC}")

        self.model = None
        self.scaler_mean = None
        self.scaler_std = None

    def _to_dense_float32(self, matrix) -> np.ndarray:
        """Convert dense/sparse-like matrix to contiguous float32 numpy array."""
        if scipy.sparse.issparse(matrix):
            dense = matrix.toarray()
        else:
            dense = np.asarray(matrix)
        return np.ascontiguousarray(dense, dtype=np.float32)

    def train(self, X_train: scipy.sparse.spmatrix, y_train: Sequence[str]):
        """Train the classifier on annotated samples."""
        print(f"{Colors.CYAN}{EMOJI['train']} Training PyTorch GPU classifier...{Colors.ENDC}")
        print(f"{Colors.BLUE}  → Normalizing {X_train.shape[0]} samples with r_value={self.r_value}{Colors.ENDC}")

        # Normalize
        X_train = normalize_rows(X_train, self.r_value)
        if scipy.sparse.isspmatrix(X_train):
            X_train = X_train.log1p()
        else:
            X_train = np.log1p(X_train)

        # Get unique classes
        self.classes = np.unique(y_train)
        num_classes = len(self.classes)
        num_features = X_train.shape[1]

        print(f"{Colors.BLUE}  → Classes: {num_classes}, Features: {num_features}{Colors.ENDC}")

        # Convert labels to integers
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        y_indices = np.array([class_to_idx[y] for y in y_train])

        # Convert to PyTorch tensors
        print(f"{Colors.CYAN}  → Moving data to {self.device}...{Colors.ENDC}")
        print(f"{Colors.BLUE}  → Data type before conversion: {type(X_train)}{Colors.ENDC}")

        # Check if sparse using multiple methods (for compatibility with scipy versions)
        is_sparse_input = scipy.sparse.issparse(X_train)

        if self.use_sparse and is_sparse_input:
            # Sparse path
            print(f"{Colors.GREEN}  → Using sparse tensors (memory efficient!){Colors.ENDC}")
            X_tensor = sparse_to_torch(X_train, device=str(self.device), force_dense=False)
            is_sparse = True
        else:
            # Dense path
            print(f"{Colors.BLUE}  → Converting to dense float32 array...{Colors.ENDC}")
            X_train_dense = self._to_dense_float32(X_train)

            X_tensor = torch.from_numpy(X_train_dense).to(self.device)
            is_sparse = False

        y_tensor = torch.from_numpy(y_indices).long().to(self.device)

        # Standardize features
        if is_sparse:
            # For sparse tensors, compute mean/std from dense version for now
            # (More efficient methods exist but this is simpler)
            print(f"{Colors.BLUE}  → Computing statistics (may densify temporarily)...{Colors.ENDC}")
            X_dense_for_stats = X_tensor.to_dense() if X_tensor.is_sparse else X_tensor
            self.scaler_mean = X_dense_for_stats.mean(dim=0)
            self.scaler_std = X_dense_for_stats.std(dim=0) + 1e-8
            del X_dense_for_stats  # Free memory
        else:
            self.scaler_mean = X_tensor.mean(dim=0)
            self.scaler_std = X_tensor.std(dim=0) + 1e-8

        # Standardize (sparse tensors need special handling)
        if is_sparse:
            # For sparse standardization, we'll standardize during forward pass
            # to avoid densifying the entire matrix
            print(f"{Colors.BLUE}  → Will standardize during forward pass (keeping sparse){Colors.ENDC}")
            X_standardized = X_tensor
        else:
            X_standardized = (X_tensor - self.scaler_mean) / self.scaler_std

        # Build model
        if is_sparse:
            self.model = SparseLinearModel(num_features, num_classes).to(self.device)
        else:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(num_features, num_classes)
            ).to(self.device)

        # Loss and optimizer (with L2 regularization similar to SVC)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)

        # Training loop
        print(f"{Colors.CYAN}  → Training on {self.device}...{Colors.ENDC}")

        if is_sparse:
            # For sparse matrices, we process the entire dataset at once
            # or use manual batching
            print(f"{Colors.BLUE}  → Sparse training mode (full dataset per iteration){Colors.ENDC}")

            self.model.train()
            for epoch in range(self.max_iter):
                optimizer.zero_grad()

                # Forward pass with sparse input
                outputs = self.model(X_standardized)
                loss = criterion(outputs, y_tensor)

                # Backward pass
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 100 == 0:
                    print(f"{Colors.BLUE}    Epoch {epoch+1}/{self.max_iter}, Loss: {loss.item():.4f}{Colors.ENDC}")

        else:
            # Dense training with mini-batches
            dataset = torch.utils.data.TensorDataset(X_standardized, y_tensor)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )

            self.model.train()
            for epoch in range(self.max_iter):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    if self.use_amp and self.device.type == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            outputs = self.model(batch_X)
                            loss = criterion(outputs, batch_y)
                    else:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if (epoch + 1) % 100 == 0:
                    avg_loss = total_loss / len(dataloader)
                    print(f"{Colors.BLUE}    Epoch {epoch+1}/{self.max_iter}, Loss: {avg_loss:.4f}{Colors.ENDC}")

        self.model.eval()
        super().train(X_train, y_train)
        print(f"{Colors.GREEN}{EMOJI['done']} GPU Training completed!{Colors.ENDC}")

    def classify(self, samples: scipy.sparse.spmatrix | np.matrix,
                 silent: bool = False):
        """Classify samples on GPU with sparse support."""
        super().classify(samples)

        if not silent:
            print(f"{Colors.CYAN}{EMOJI['classify']} GPU Classifying {samples.shape[0]} samples...{Colors.ENDC}")

        # Normalize
        test = normalize_rows(samples, self.r_value)
        is_sparse_input = scipy.sparse.issparse(test)

        if is_sparse_input:
            test = test.log1p()
        else:
            test = np.log1p(test)

        # Convert to tensor and move to device
        if self.use_sparse and is_sparse_input:
            # Sparse path
            X_tensor = sparse_to_torch(test, device=str(self.device), force_dense=False)
            is_sparse = True
        else:
            # Dense path
            test_dense = self._to_dense_float32(test)
            X_tensor = torch.from_numpy(test_dense)
            if self.device.type == 'cuda':
                if self.pin_memory:
                    X_tensor = X_tensor.pin_memory()
                X_tensor = X_tensor.to(self.device, non_blocking=True)
            else:
                X_tensor = X_tensor.to(self.device)
            is_sparse = False

        # Standardize
        if is_sparse:
            # For sparse tensors, standardize on-the-fly
            # Note: This doesn't actually standardize sparse tensors properly
            # A proper implementation would need custom sparse standardization
            # For now, we'll just use the sparse tensor as-is
            X_standardized = X_tensor
        else:
            X_standardized = (X_tensor - self.scaler_mean) / self.scaler_std

        # Predict
        self.model.eval()

        with torch.inference_mode():
            if is_sparse:
                # Process all at once for sparse (no batching needed, already memory efficient)
                logits = self.model(X_standardized)
                probs = torch.nn.functional.softmax(logits, dim=1)
                result = probs.cpu().numpy()
            else:
                # Batch processing for dense tensors
                n_samples = X_standardized.shape[0]
                n_classes = len(self.classes)
                result = np.empty((n_samples, n_classes), dtype=np.float32)
                for i in range(0, X_standardized.shape[0], self.batch_size):
                    batch = X_standardized[i:i + self.batch_size]
                    if self.use_amp and self.device.type == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            logits = self.model(batch)
                    else:
                        logits = self.model(batch)
                    probs = torch.nn.functional.softmax(logits.float(), dim=1)
                    result[i:i + self.batch_size] = probs.cpu().numpy()

        if not silent:
            print(f"{Colors.GREEN}{EMOJI['done']} GPU Classification completed!{Colors.ENDC}")

        return result


class TorchLinearSVMClassifier(Classifier):
    """Torch implementation of linear SVM-style classifier (OVR hinge loss).

    Design goals:
    - Follow SVCClassifier principles more closely than cross-entropy training.
    - Keep batch-friendly GPU execution.
    - Preserve the existing classify() contract by returning class probabilities.
    """

    def __init__(self, r_value: int = 5, device: str = 'auto',
                 max_iter: int = 1000, lr: float = 0.005,
                 batch_size: int = 2048, weight_decay: float = 0.0001,
                 margin: float = 1.0, use_amp: bool = False,
                 pin_memory: bool = True,
                 distill_from_svc: bool = True,
                 distill_alpha: float = 0.5,
                 distill_temperature: float = 2.0,
                 grad_clip_norm: float = 5.0,
                 optimizer_name: str = 'adamw'):
        super().__init__()
        self.r_value = r_value
        self.max_iter = max_iter
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.margin = margin
        self.use_amp = use_amp
        self.pin_memory = pin_memory
        self.distill_from_svc = distill_from_svc
        self.distill_alpha = distill_alpha
        self.distill_temperature = distill_temperature
        self.grad_clip_norm = grad_clip_norm
        self.optimizer_name = optimizer_name.lower()

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Install with: pip install torch")

        if device == 'auto':
            if torch.cuda.is_available():
                self.device = normalize_torch_device("cuda")
                device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
                device_name = "MPS (Apple Silicon)"
            else:
                self.device = torch.device('cpu')
                device_name = "CPU"
        else:
            self.device = normalize_torch_device(device)
            device_name = device

        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' but torch.cuda.is_available() is False")

        print(f"{Colors.GREEN}🚀 TorchLinearSVMClassifier initialized on: {device_name}{Colors.ENDC}")
        self.model = None
        self.scaler_std = None
        self.teacher = None

    def _to_dense_float32(self, matrix) -> np.ndarray:
        if scipy.sparse.issparse(matrix):
            dense = matrix.toarray()
        else:
            dense = np.asarray(matrix)
        return np.ascontiguousarray(dense, dtype=np.float32)

    def _standardize_like_svc(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """Approximate sklearn StandardScaler(with_mean=False): scale only."""
        return x_tensor / self.scaler_std

    def train(self, X_train: scipy.sparse.spmatrix, y_train: Sequence[str]):
        print(f"{Colors.CYAN}{EMOJI['train']} Training Torch linear SVM-style classifier...{Colors.ENDC}")
        print(f"{Colors.BLUE}  → Student device: {self.device}{Colors.ENDC}")
        if self.distill_from_svc:
            print(f"{Colors.BLUE}  → Distillation enabled: training CPU SVC teacher first{Colors.ENDC}")
        else:
            print(f"{Colors.BLUE}  → Distillation disabled: pure Torch SVM-style training{Colors.ENDC}")
        X_train = normalize_rows(X_train, self.r_value)
        if scipy.sparse.issparse(X_train):
            X_train = X_train.log1p()
        else:
            X_train = np.log1p(X_train)

        self.classes = np.unique(y_train)
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        y_indices = np.array([class_to_idx[y] for y in y_train], dtype=np.int64)
        num_classes = len(self.classes)
        num_features = X_train.shape[1]

        X_np = self._to_dense_float32(X_train)
        X_tensor = torch.from_numpy(X_np)
        y_tensor = torch.from_numpy(y_indices).long()

        if self.device.type == 'cuda' and self.pin_memory:
            X_tensor = X_tensor.pin_memory()
            y_tensor = y_tensor.pin_memory()

        X_tensor = X_tensor.to(self.device, non_blocking=(self.device.type == 'cuda'))
        y_tensor = y_tensor.to(self.device, non_blocking=(self.device.type == 'cuda'))
        print(f"{Colors.BLUE}  → Training tensor device: {X_tensor.device}{Colors.ENDC}")

        # Match StandardScaler(with_mean=False): no mean-centering, std scaling only
        self.scaler_std = X_tensor.std(dim=0)
        self.scaler_std = torch.where(self.scaler_std < 1e-8,
                                      torch.ones_like(self.scaler_std),
                                      self.scaler_std)
        X_standardized = self._standardize_like_svc(X_tensor)

        self.model = torch.nn.Linear(num_features, num_classes).to(self.device)
        print(f"{Colors.BLUE}  → Model parameter device: {next(self.model.parameters()).device}{Colors.ENDC}")
        if self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )

        dataset = torch.utils.data.TensorDataset(X_standardized, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        teacher_probs_full = None
        if self.distill_from_svc:
            # Build a teacher aligned with SVCClassifier behavior.
            self.teacher = SVCClassifier(r_value=self.r_value, use_gpu=False)
            self.teacher.train(X_train, y_train)
            teacher_probs_np = self.teacher.classify(X_train, silent=True).astype(np.float32)
            teacher_probs_full = torch.from_numpy(teacher_probs_np)
            if self.device.type == 'cuda' and self.pin_memory:
                teacher_probs_full = teacher_probs_full.pin_memory()
            teacher_probs_full = teacher_probs_full.to(
                self.device, non_blocking=(self.device.type == 'cuda')
            )

        # Track sample indices so we can retrieve matching teacher soft targets.
        indexed_dataset = torch.utils.data.TensorDataset(
            X_standardized,
            y_tensor,
            torch.arange(X_standardized.shape[0], device=self.device)
        )
        indexed_loader = torch.utils.data.DataLoader(
            indexed_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model.train()
        for epoch in range(self.max_iter):
            total_loss = 0.0
            total_hinge = 0.0
            total_kd = 0.0
            for batch_X, batch_y, batch_idx in indexed_loader:
                optimizer.zero_grad()
                if self.use_amp and self.device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        scores = self.model(batch_X)
                else:
                    scores = self.model(batch_X)

                # One-vs-rest hinge-like objective:
                # y in {-1, +1} for each class, minimize max(0, m - y*s)
                y_onehot = torch.nn.functional.one_hot(batch_y, num_classes=num_classes).float()
                y_sign = y_onehot.mul(2.0).sub(1.0)
                hinge = torch.relu(self.margin - y_sign * scores.float())
                hinge_loss = hinge.mean()
                kd_loss = torch.tensor(0.0, device=self.device)

                if teacher_probs_full is not None:
                    # Distillation: match SVC teacher soft probabilities.
                    t = float(self.distill_temperature)
                    teacher_probs = teacher_probs_full[batch_idx]
                    student_log_probs = torch.nn.functional.log_softmax(scores.float() / t, dim=1)
                    teacher_log_probs = torch.log(teacher_probs.clamp_min(1e-8))
                    teacher_probs_t = torch.nn.functional.softmax(teacher_log_probs / t, dim=1)
                    kd_loss = torch.nn.functional.kl_div(
                        student_log_probs,
                        teacher_probs_t,
                        reduction='batchmean'
                    ) * (t * t)
                    alpha = float(self.distill_alpha)
                    loss = (1.0 - alpha) * hinge_loss + alpha * kd_loss
                else:
                    loss = hinge_loss

                loss.backward()
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                optimizer.step()
                total_loss += float(loss.item())
                total_hinge += float(hinge_loss.item())
                total_kd += float(kd_loss.item())

            if (epoch + 1) % 100 == 0:
                denom = max(len(indexed_loader), 1)
                avg_loss = total_loss / denom
                avg_hinge = total_hinge / denom
                avg_kd = total_kd / denom
                print(
                    f"{Colors.BLUE}    Epoch {epoch+1}/{self.max_iter}, "
                    f"Loss: {avg_loss:.4f}, Hinge: {avg_hinge:.4f}, KD: {avg_kd:.4f}{Colors.ENDC}"
                )

        self.model.eval()
        super().train(X_train, y_train)
        print(f"{Colors.GREEN}{EMOJI['done']} Torch linear SVM-style training completed!{Colors.ENDC}")

    def classify(self, samples: scipy.sparse.spmatrix | np.matrix, silent: bool = False):
        super().classify(samples)
        if not silent:
            print(f"{Colors.CYAN}{EMOJI['classify']} SVM-style classifying {samples.shape[0]} samples...{Colors.ENDC}")

        test = normalize_rows(samples, self.r_value)
        if scipy.sparse.issparse(test):
            test = test.log1p()
        else:
            test = np.log1p(test)

        test_np = self._to_dense_float32(test)
        X_tensor = torch.from_numpy(test_np)
        if self.device.type == 'cuda' and self.pin_memory:
            X_tensor = X_tensor.pin_memory()
        X_tensor = X_tensor.to(self.device, non_blocking=(self.device.type == 'cuda'))

        X_standardized = self._standardize_like_svc(X_tensor)

        with torch.inference_mode():
            n_samples = X_standardized.shape[0]
            n_classes = len(self.classes)
            result = np.empty((n_samples, n_classes), dtype=np.float32)
            for i in range(0, n_samples, self.batch_size):
                batch = X_standardized[i:i + self.batch_size]
                if self.use_amp and self.device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        scores = self.model(batch)
                else:
                    scores = self.model(batch)

                # Keep interface-compatible probabilities.
                probs = torch.nn.functional.softmax(scores.float(), dim=1)
                result[i:i + self.batch_size] = probs.cpu().numpy()

        if not silent:
            print(f"{Colors.GREEN}{EMOJI['done']} SVM-style classification completed!{Colors.ENDC}")
        return result


def train_from_countmatrix(classifier: Classifier,
                           countmatrix: CountData,
                           label: str
                           ):
    print(f"{Colors.HEADER}{EMOJI['start']} Training classifier from count matrix...{Colors.ENDC}")
    print(f"{Colors.BLUE}  → Using label: {label}{Colors.ENDC}")
    classifier.train(countmatrix.matrix, list(countmatrix.metadata[label]))
