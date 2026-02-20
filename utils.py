import numpy as np
import scipy.sparse
import torch
import ot
from typing import Any, Optional
from anndata import AnnData
from numpy.typing import NDArray

def extract_data_matrix(adata: AnnData, use_rep: Optional[str] = None) -> NDArray[Any]:
    if use_rep is None:
        return adata.X
    return adata.obsm[use_rep]

def to_dense_array(X) -> NDArray[Any]:
    if scipy.sparse.issparse(X):
        return X.toarray()
    return np.array(X)

def to_backend_array(x: Any, nx: Any, dtype: str = "float64", use_gpu: bool = True) -> Any:
    """Convert input to backend array (NumPy or Torch), handle device placement."""
    if isinstance(x, np.ndarray):
        x = nx.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        if not isinstance(nx, ot.backend.TorchBackend):
            x = x.detach().cpu().numpy()
            x = nx.from_numpy(x)
    if isinstance(nx, ot.backend.TorchBackend):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.float32
        x = x.to(dtype)
        device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        x = x.to(device)
    return x

def to_numpy(x) -> NDArray[Any]:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif hasattr(x, '__array__'):
        return np.array(x)
    nx = ot.backend.get_backend(x)
    return nx.to_numpy(x)
