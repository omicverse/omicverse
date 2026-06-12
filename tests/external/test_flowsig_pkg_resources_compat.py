import builtins
import sys
import types

import anndata as ad
import numpy as np
import pandas as pd

from omicverse.external.flowsig.preprocessing._flow_preprocessing import (
    determine_spatially_flowing_vars,
)


def test_flowsig_pkg_resources_compat(monkeypatch):
    pkg_resources_was_absent = "pkg_resources" not in sys.modules
    monkeypatch.delitem(sys.modules, "pkg_resources", raising=False)
    monkeypatch.delitem(sys.modules, "squidpy", raising=False)
    observed = {}

    fake_squidpy = types.SimpleNamespace()

    def spatial_neighbors(adata, **_kwargs):
        adata.obsp["spatial_connectivities"] = np.eye(adata.n_obs)

    def spatial_autocorr(adata, genes, **_kwargs):
        adata.uns["moranI"] = pd.DataFrame(
            {"I": [0.2 for _ in genes]},
            index=genes,
        )

    fake_squidpy.gr = types.SimpleNamespace(
        spatial_neighbors=spatial_neighbors,
        spatial_autocorr=spatial_autocorr,
    )

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "squidpy":
            import pkg_resources

            observed["pkg_resources_ready"] = "pkg_resources" in sys.modules
            observed["require"] = pkg_resources.require("setuptools")
            observed["working_set"] = list(pkg_resources.working_set)
            observed["entry_points"] = list(pkg_resources.iter_entry_points("console_scripts"))
            observed["setuptools_version"] = pkg_resources.get_distribution(
                "setuptools"
            ).version
            observed["omicverse_init"] = pkg_resources.resource_filename(
                "omicverse", "__init__.py"
            )
            return fake_squidpy
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        adata = ad.AnnData(np.ones((4, 2)))
        adata.obsm["spatial"] = np.arange(8).reshape(4, 2)
        adata.obsm["X_flow"] = np.ones((4, 3))
        adata.uns["flowsig_network"] = {
            "flow_var_info": pd.DataFrame(
                {"Type": ["inflow", "module", "outflow"]},
                index=["inflow_a", "GEM-1", "outflow_a"],
            )
        }

        determine_spatially_flowing_vars(adata, moran_threshold=0.1)

        assert observed["pkg_resources_ready"]
        assert isinstance(observed["require"], list)
        assert isinstance(observed["working_set"], list)
        assert isinstance(observed["entry_points"], list)
        assert observed["setuptools_version"].split(".")[0].isdigit()
        assert observed["omicverse_init"].endswith("omicverse/__init__.py")
        assert list(adata.uns["flowsig_network"]["flow_var_info"].index) == [
            "inflow_a",
            "GEM-1",
            "outflow_a",
        ]
    finally:
        if pkg_resources_was_absent:
            sys.modules.pop("pkg_resources", None)
