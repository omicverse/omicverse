from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

# Preload the exact PyG modules used by the vendored layers before temporarily
# blocking torch_sparse, so this test isolates the backends' optional import.
from torch_geometric.nn.conv import MessagePassing  # noqa: F401, E402
from torch_geometric.typing import Adj  # noqa: F401, E402
from torch_geometric.utils import add_self_loops  # noqa: F401, E402


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_without_torch_sparse(monkeypatch, name: str, relative_path: str):
    path = _REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "torch_sparse", None)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("module_name", "relative_path", "backend_name"),
    [
        (
            "_test_staligner_gat_without_torch_sparse",
            "omicverse/external/STAligner/gat_conv.py",
            "STAligner",
        ),
        (
            "_test_binary_gat_without_torch_sparse",
            "omicverse/external/BINARY/Model.py",
            "BINARY",
        ),
    ],
)
def test_tensor_edge_index_does_not_require_torch_sparse(
    monkeypatch, module_name, relative_path, backend_name
):
    module = _load_without_torch_sparse(monkeypatch, module_name, relative_path)

    x = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0], [1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 0]],
        dtype=torch.long,
    )
    layer = module.GATConv(
        in_channels=3,
        out_channels=2,
        heads=1,
        concat=False,
        add_self_loops=True,
        bias=False,
    )

    output = layer(x, edge_index)

    assert output.shape == (4, 2)
    assert torch.isfinite(output).all()
    with pytest.raises(ImportError, match=rf"SparseTensor {backend_name} inputs"):
        layer(x, object())
