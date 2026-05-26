from typing import Dict, Hashable, Literal, Optional, Protocol, Sequence, Union

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

MatrixLike = Union[np.ndarray, spmatrix]


class AnnDataLayersLike(Protocol):
    """Layer container surface used by AnnData-like inputs."""

    def __getitem__(self, key: str) -> MatrixLike:
        ...


class AnnDataLike(Protocol):
    """Structural type for AnnData-compatible expression objects."""

    X: MatrixLike
    layers: AnnDataLayersLike
    var_names: Sequence[Hashable]
    obs_names: Sequence[Hashable]


ExpressionData = Union[pd.DataFrame, AnnDataLike]
LayerName = Optional[str]
Kwargs = Dict[str, object]
Backend = Literal["serial", "joblib-loky", "joblib-threading", "ray"]
KOMethod = Literal["default", "propagation"]
