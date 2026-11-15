from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import time

from .._registry import register_function
from .._settings import Colors, EMOJI
from ..report._provenance import tracked, note


class _gsmap_runner:
    """Small gsMap workflow runner for OmicVerse spatial workflows."""

    def __init__(self, adata, *, workdir: str, sample_name: str, annotation: str | None = None):
        self.adata = adata
        self.workdir = workdir
        self.sample_name = sample_name
        self.annotation = annotation

    @property
    def hdf5_with_latent_path(self) -> Path:
        return (
            Path(self.workdir)
            / self.sample_name
            / "find_latent_representations"
            / f"{self.sample_name}_add_latent.h5ad"
        )

    @property
    def mkscore_feather_path(self) -> Path:
        return (
            Path(self.workdir)
            / self.sample_name
            / "latent_to_gene"
            / f"{self.sample_name}_gene_marker_score.feather"
        )

    def prepare_input_h5ad(self) -> Path:
        """Write the current AnnData object to a gsMap-style input location."""

        target_dir = Path(self.workdir) / self.sample_name / "input"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{self.sample_name}.h5ad"
        self.adata.write_h5ad(target_path)
        return target_path

    def build_find_latent_config(self, **overrides):
        """Build the config object for the gsMap latent-representation step."""

        from ..external.gsmap import FindLatentRepresentationsConfig

        input_path = self.prepare_input_h5ad()
        return FindLatentRepresentationsConfig(
            workdir=self.workdir,
            sample_name=self.sample_name,
            input_hdf5_path=str(input_path),
            annotation=self.annotation,
            **overrides,
        )

    def build_latent_to_gene_config(self, **overrides):
        """Build the config object for the gsMap latent-to-gene step."""

        from ..external.gsmap import LatentToGeneConfig

        return LatentToGeneConfig(
            workdir=self.workdir,
            sample_name=self.sample_name,
            annotation=self.annotation,
            **overrides,
        )

    def build_generate_ldscore_config(self, **overrides):
        """Build the config object for the gsMap generate-ldscore step."""

        from ..external.gsmap import GenerateLdscoreConfig

        gsmap_resource_dir = Path(overrides.pop("gsmap_resource_dir"))
        quick_mode_dir = gsmap_resource_dir / "quick_mode"

        return GenerateLdscoreConfig(
            workdir=self.workdir,
            sample_name=self.sample_name,
            baseline_annotation_dir=quick_mode_dir / "baseline",
            snp_gene_pair_dir=quick_mode_dir / "SNP_gene_pair",
            **overrides,
        )

    @tracked('find_latent_representation', 'ov.genetics.gsmap.find_latent_representation',
             adata_attr='adata')
    def find_latent_representation(self, **overrides) -> Path:
        """Run the gsMap latent-representation step and return its output path.

        Parameters
        ----------
        **overrides
            Keyword arguments forwarded to :meth:`build_find_latent_config`.

        Returns
        -------
        pathlib.Path
            Path to the latent-augmented H5AD file.
        """

        from ..external.gsmap import run_find_latent_representation
        from ..external.gsmap._style import ov_print_done, ov_print_start, ov_print_time

        ov_print_start("find_latent_representation")
        t0 = time()
        config = self.build_find_latent_config(**overrides)
        run_find_latent_representation(config)
        note(backend='omicverse · gsMap')
        ov_print_time(time() - t0)
        ov_print_done("find_latent_representation")
        return config.hdf5_with_latent_path

    @tracked('latent_to_gene', 'ov.genetics.gsmap.latent_to_gene',
             adata_attr='adata')
    def latent_to_gene(self, **overrides) -> Path:
        """Run the gsMap latent-to-gene step and return its feather output path.

        This step maps the latent representation back to gene-level specificity
        scores (GSS) and writes a ``.feather`` file consumed by
        :meth:`generate_ldscore`.

        Parameters
        ----------
        input_hdf5_path : str, optional
            Path to the latent-augmented h5ad file. Defaults to the output of
            :meth:`find_latent_representation`.
        latent_representation : str, optional
            Key in ``adata.obsm`` that stores the latent embedding
            (e.g. ``'latent_GVAE'``).
        num_neighbour : int, default 51
            Number of nearest neighbours in the latent space for smoothing.
        num_neighbour_spatial : int, default 201
            Number of nearest neighbours in the spatial coordinate space.
        species : str, optional
            Species identifier for homolog transformation.
            **Required for non-human ST data** (e.g. ``'MOUSE_GENE_SYM'``).
            **Omit for human ST data** — gene symbols are already human.
        homolog_file : str, optional
            Path to a two-column TSV mapping ``species`` gene symbols to human
            gene symbols (``HUMAN_GENE_SYM``).
            **Required when ``species`` is provided.**
            **Omit for human ST data.**
        annotation : str, optional
            ``adata.obs`` column used for cell-type aware aggregation.
        no_expression_fraction : bool, default False
            If ``True``, skip expression-fraction filtering.
        gM_slices : str, optional
            Spatial slice identifier for multi-slice datasets.

        Returns
        -------
        pathlib.Path
            Path to the marker-score feather file
            (``{sample_name}_gene_marker_score.feather``).

        Examples
        --------
        >>> # Human ST — no homolog conversion needed
        >>> marker_path = runner.latent_to_gene(
        ...     latent_representation='latent_GVAE',
        ...     num_neighbour=51,
        ...     num_neighbour_spatial=201,
        ... )
        >>> # Mouse ST — must provide homolog mapping
        >>> marker_path = runner.latent_to_gene(
        ...     latent_representation='latent_GVAE',
        ...     num_neighbour=51,
        ...     num_neighbour_spatial=201,
        ...     species='MOUSE_GENE_SYM',
        ...     homolog_file='/path/to/mouse_human_homologs.txt',
        ... )
        """

        from ..external.gsmap import run_latent_to_gene
        from ..external.gsmap._style import ov_print_done, ov_print_start, ov_print_time

        ov_print_start("latent_to_gene")
        t0 = time()
        config = self.build_latent_to_gene_config(**overrides)
        run_latent_to_gene(config)
        note(backend='omicverse · gsMap')
        ov_print_time(time() - t0)
        ov_print_done("latent_to_gene")
        return config.mkscore_feather_path

    @tracked('generate_ldscore', 'ov.genetics.gsmap.generate_ldscore',
             adata_attr='adata')
    def generate_ldscore(self, **overrides) -> Path:
        """Run the gsMap generate-ldscore step and return its output directory.

        Parameters
        ----------
        **overrides
            Keyword arguments forwarded to :meth:`build_generate_ldscore_config`.

        Returns
        -------
        pathlib.Path
            Path to the LD-score output directory.
        """

        from ..external.gsmap import run_generate_ldscore
        from ..external.gsmap._style import ov_print_done, ov_print_start, ov_print_time

        ov_print_start("generate_ldscore")
        t0 = time()
        config = self.build_generate_ldscore_config(**overrides)
        run_generate_ldscore(config)
        note(backend='omicverse · gsMap')
        ov_print_time(time() - t0)
        ov_print_done("generate_ldscore")
        return config.ldscore_save_dir

    def build_spatial_ldsc_config(self, **overrides):
        """Build the config object for the gsMap spatial-LDSC step."""

        from ..external.gsmap import SpatialLdscConfig

        gsmap_resource_dir = Path(overrides.pop("gsmap_resource_dir"))

        return SpatialLdscConfig(
            workdir=self.workdir,
            sample_name=self.sample_name,
            w_file=gsmap_resource_dir / "LDSC_resource" / "weights_hm3_no_hla" / "weights.",
            snp_gene_weight_adata_path=gsmap_resource_dir / "quick_mode" / "snp_gene_weight_matrix.h5ad",
            **overrides,
        )

    @tracked('spatial_ldsc', 'ov.genetics.gsmap.spatial_ldsc',
             adata_attr='adata')
    def spatial_ldsc(self, **overrides) -> Path:
        """Run the gsMap spatial-LDSC step and return its output directory.

        After completion, p-values are automatically written back to the latent
        ``adata`` as ``{trait}_gsmap_p`` (raw p) and ``{trait}_gsmap_logp``
        (``-log10(p)``) and saved to the latent h5ad file.

        Parameters
        ----------
        **overrides
            Keyword arguments forwarded to :meth:`build_spatial_ldsc_config`.

        Returns
        -------
        pathlib.Path
            Path to the spatial-LDSC output directory.
        """

        from ..external.gsmap import run_spatial_ldsc
        from ..external.gsmap._style import ov_print_done, ov_print_start, ov_print_time
        import numpy as np
        import pandas as pd

        ov_print_start("spatial_ldsc")
        t0 = time()
        config = self.build_spatial_ldsc_config(**overrides)
        run_spatial_ldsc(config)

        # Write LDSC p-values back to latent adata
        adata = self._get_latent_adata()
        for trait_name in config.sumstats_config_dict.keys():
            ldsc_path = config.ldsc_save_dir / f"{config.sample_name}_{trait_name}.csv.gz"
            if not ldsc_path.exists():
                continue
            ldsc = pd.read_csv(ldsc_path, compression="gzip", dtype={"spot": str, "p": float})
            ldsc.set_index("spot", inplace=True)
            ldsc["logp"] = -np.log10(ldsc["p"].clip(lower=1e-300))
            adata.obs[f"{trait_name}_gsmap_p"] = ldsc.reindex(adata.obs_names)["p"]
            adata.obs[f"{trait_name}_gsmap_logp"] = ldsc.reindex(adata.obs_names)["logp"]
        latent_path = (
            Path(self.workdir) / self.sample_name / "find_latent_representations"
            / f"{self.sample_name}_add_latent.h5ad"
        )
        adata.write_h5ad(latent_path)

        note(backend='omicverse · gsMap')
        ov_print_time(time() - t0)
        ov_print_done("spatial_ldsc")
        return config.ldsc_save_dir

    def build_cauchy_combination_config(self, **overrides):
        """Build the config object for the gsMap cauchy-combination step."""

        from ..external.gsmap import CauchyCombinationConfig

        annotation = overrides.pop("annotation", self.annotation)
        return CauchyCombinationConfig(
            workdir=self.workdir,
            sample_name=self.sample_name,
            annotation=annotation,
            **overrides,
        )

    @tracked('cauchy_combination', 'ov.genetics.gsmap.cauchy_combination',
             adata_attr='adata')
    def cauchy_combination(self, **overrides) -> Path:
        """Run the gsMap cauchy-combination step and return its result file.

        Parameters
        ----------
        **overrides
            Keyword arguments forwarded to :meth:`build_cauchy_combination_config`.

        Returns
        -------
        pathlib.Path
            Path to the Cauchy-combination CSV file.
        """

        from ..external.gsmap import run_cauchy_combination
        from ..external.gsmap._style import ov_print_done, ov_print_start, ov_print_time

        ov_print_start("cauchy_combination")
        t0 = time()
        config = self.build_cauchy_combination_config(**overrides)
        run_cauchy_combination(config)
        note(backend='omicverse · gsMap')
        ov_print_time(time() - t0)
        ov_print_done("cauchy_combination")
        return config.output_file

    def build_report_config(self, **overrides):
        """Build the config object for the gsMap final report step."""

        from ..external.gsmap import ReportConfig

        annotation = overrides.pop("annotation", self.annotation)
        return ReportConfig(
            workdir=self.workdir,
            sample_name=self.sample_name,
            annotation=annotation,
            **overrides,
        )

    @tracked('report', 'ov.genetics.gsmap.report',
             adata_attr='adata')
    def report(self, **overrides) -> Path:
        """Run the gsMap final report step and return its HTML report path.

        Parameters
        ----------
        **overrides
            Keyword arguments forwarded to :meth:`build_report_config`.

        Returns
        -------
        pathlib.Path
            Path to the generated HTML report.
        """

        from ..external.gsmap import run_report
        from ..external.gsmap._style import ov_print_done, ov_print_start, ov_print_time

        ov_print_start("report")
        t0 = time()
        config = self.build_report_config(**overrides)
        run_report(config)
        note(backend='omicverse · gsMap')
        ov_print_time(time() - t0)
        ov_print_done("report")
        return config.get_gsmap_report_file(config.trait_name)

    def _get_ldsc_result_path(self, trait_name: str) -> Path:
        """Return the spatial-LDSC result CSV path for a trait."""
        return (
            Path(self.workdir)
            / self.sample_name
            / "spatial_ldsc"
            / f"{self.sample_name}_{trait_name}.csv.gz"
        )

    def _get_cauchy_result_path(self, trait_name: str) -> Path:
        """Return the cauchy-combination result CSV path for a trait."""
        return (
            Path(self.workdir)
            / self.sample_name
            / "cauchy_combination"
            / f"{self.sample_name}_{trait_name}.Cauchy.csv.gz"
        )

    def _get_latent_adata(self):
        """Load the latent-augmented AnnData used by downstream plotting."""
        import anndata

        latent_path = (
            Path(self.workdir)
            / self.sample_name
            / "find_latent_representations"
            / f"{self.sample_name}_add_latent.h5ad"
        )
        return anndata.read_h5ad(latent_path)

    def plot_manhattan(
        self,
        trait_name: str,
        sumstats_file: str,
        top_corr_genes: int = 50,
        save_path: str | Path | None = None,
        show: bool = True,
        **kwargs,
    ):
        """Generate the gsMap diagnosis Manhattan plot.

        This reuses the existing ``manhattan_plot`` logic from gsMap's vendored
        code, but returns the figure directly instead of embedding it in HTML.

        Parameters
        ----------
        trait_name : str
            Trait name for the diagnosis plot.
        sumstats_file : str
            Path to the GWAS summary-statistics file.
        top_corr_genes : int, default 50
            Number of top-correlated genes to highlight.
        save_path : str or Path, optional
            Path to save the figure (.png or .html).
        show : bool, default True
            Whether to display the figure.
        **kwargs
            Additional arguments forwarded to ``manhattan_plot``.

        Returns
        -------
        plotly.graph_objects.Figure
            The generated Manhattan plot figure.
        """
        import numpy as np
        import pandas as pd
        from scipy.stats import norm

        from ..pp import normalize_total, log1p

        from ..external.gsmap._manhattan_plot import manhattan_plot
        from ..external.gsmap.diagnosis import compute_gene_diagnostic_info, load_gene_diagnostic_info
        from ..external.gsmap._config import DiagnosisConfig

        config = DiagnosisConfig(
            workdir=self.workdir,
            sample_name=self.sample_name,
            annotation=self.annotation or "annotation",
            trait_name=trait_name,
            sumstats_file=sumstats_file,
            top_corr_genes=top_corr_genes,
        )

        adata = self._get_latent_adata()
        if "log1p" not in adata.uns and adata.X.max() > 14:
            normalize_total(adata, target_sum=1e4)
            log1p(adata)

        gene_info = load_gene_diagnostic_info(config, adata)
        if gene_info.empty:
            gene_info = compute_gene_diagnostic_info(config, adata)

        gwas_data = pd.read_csv(sumstats_file, compression="gzip", sep="\t")
        gwas_data["P"] = norm.sf(abs(gwas_data["Z"])) * 2
        gwas_data["P"] = gwas_data["P"].clip(lower=1e-300)

        snp_gene_pair_prefix = config.ldscore_save_dir / "SNP_gene_pair" / "SNP_gene_pair_chr"
        from ..external.gsmap._regression_read import _read_chr_files

        snp_gene_pair = pd.concat(
            [
                pd.read_feather(file_path)
                for file_path in _read_chr_files(snp_gene_pair_prefix.as_posix(), suffix=".feather")
            ]
        )

        gwas_data_with_gene = snp_gene_pair.merge(gwas_data, on="SNP", how="inner").rename(
            columns={"gene_name": "GENE"}
        )
        merged = gwas_data_with_gene.merge(gene_info, left_on="GENE", right_on="Gene", how="left")
        merged = merged[~merged["Annotation"].isna()]
        merged = merged.sort_values("P")

        if merged.empty:
            raise ValueError("Filtered GWAS data is empty, cannot create Manhattan plot.")

        pass_suggestive_line_mask = merged["P"] < 1e-5
        pass_suggestive_line_number = pass_suggestive_line_mask.sum()
        if pass_suggestive_line_number > 100_000:
            snps_to_plot = merged[pass_suggestive_line_mask].SNP
        else:
            snps_to_plot = merged.head(100_000).SNP
        plot_df = merged[merged["SNP"].isin(snps_to_plot)].reset_index(drop=True)

        plot_df["Annotation_text"] = (
            "PCC: "
            + plot_df["PCC"].round(2).astype(str)
            + "<br>Annotation: "
            + plot_df["Annotation"].astype(str)
        )

        fig = manhattan_plot(
            dataframe=plot_df,
            title=f"{trait_name} gsMap Diagnosis Manhattan Plot",
            point_size=3,
            highlight_gene_list=gene_info.Gene.iloc[:5].tolist(),
            suggestiveline_value=-np.log10(1e-5),
            annotation="Annotation_text",
            **kwargs,
        )

        # Post-process: non-italic Arial x-tick numbers, legend moved to top
        fig.update_xaxes(tickfont=dict(family="Arial", size=12))
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        if save_path is not None:
            save_path_str = str(save_path)
            if save_path_str.endswith(".html"):
                fig.write_html(save_path_str)
                print(f"{EMOJI['done']} Saved Manhattan plot as HTML to {save_path}")
            else:
                try:
                    fig.write_image(save_path_str, scale=3)
                    print(f"{EMOJI['done']} Saved Manhattan plot to {save_path}")
                except ValueError as exc:
                    if "kaleido" in str(exc).lower():
                        html_path = save_path_str.replace(".png", ".html")
                        fig.write_html(html_path)
                        print(
                            f"{Colors.WARNING}⚠️  kaleido not installed; saved Manhattan plot as HTML to {html_path}{Colors.ENDC}"
                        )
                    else:
                        raise
        if show:
            fig.show()
        return fig

    def plot_gene_gss(
        self,
        trait_name: str,
        gene: str | list[str],
        save_dir: str | Path | None = None,
        size: float = 50,
        show: bool = True,
        ncols: int = 2,
        **kwargs,
    ):
        """Plot expression and GSS distribution for one or multiple genes on spatial coordinates.

        Parameters
        ----------
        trait_name : str
            Trait name used to locate marker scores.
        gene : str or list of str
            Gene(s) to plot. If a list is provided, all genes are drawn in one
            figure using ``ov.pl.embedding`` with ``ncols``.
        save_dir : str or Path, optional
            Directory to save PNG figures. When multiple genes are plotted,
            one combined figure is saved.
        size : float, default 50
            Point size for spatial dots.
        show : bool, default True
            Whether to display the figure(s).
        ncols : int, default 2
            Number of columns when plotting multiple genes (each gene produces
            an expression panel and a GSS panel).
        **kwargs
            Additional arguments forwarded to ``ov.pl.embedding``.

        Returns
        -------
        matplotlib.axes.Axes or list of Axes
            When ``gene`` is a single string, returns ``(expr_ax, gss_ax)``.
            When ``gene`` is a list, returns the list of Axes produced by
            ``ov.pl.embedding``.
        """
        import numpy as np
        import pandas as pd
        import omicverse as ov

        mkscore_path = (
            Path(self.workdir)
            / self.sample_name
            / "latent_to_gene"
            / f"{self.sample_name}_gene_marker_score.feather"
        )
        if not mkscore_path.exists():
            raise FileNotFoundError(f"Marker score not found: {mkscore_path}")

        gene_list = [gene] if isinstance(gene, str) else list(gene)

        adata_plot = self._get_latent_adata()
        for g in gene_list:
            if g not in adata_plot.var_names:
                raise ValueError(f"Gene {g} not found in adata.var_names")

        mk_score = pd.read_feather(mkscore_path).set_index("HUMAN_GENE_SYM").T
        common = adata_plot.obs_names.intersection(mk_score.index)
        adata_plot = adata_plot[common].copy()
        mk_score = mk_score.loc[common]

        color_keys = []
        for g in gene_list:
            expr = adata_plot[:, g].X
            if hasattr(expr, "toarray"):
                expr = expr.toarray()
            adata_plot.obs[f"{g}_expr"] = np.asarray(expr, dtype=np.float64).flatten()
            adata_plot.obs[f"{g}_gss"] = mk_score[g].reindex(adata_plot.obs_names).values
            color_keys.extend([f"{g}_expr", f"{g}_gss"])

        # Single gene: keep original two-figure behaviour for backward compat
        if isinstance(gene, str):
            ax_expr = ov.pl.embedding(
                adata_plot,
                basis="spatial",
                color=f"{gene}_expr",
                cmap="Reds",
                title=f"{gene} Expression",
                size=size,
                show=False,
                **kwargs,
            )
            fig_expr = ax_expr.figure if hasattr(ax_expr, "figure") else None

            ax_gss = ov.pl.embedding(
                adata_plot,
                basis="spatial",
                color=f"{gene}_gss",
                cmap="YlOrRd",
                title=f"{gene} GSS",
                size=size,
                show=False,
                **kwargs,
            )
            fig_gss = ax_gss.figure if hasattr(ax_gss, "figure") else None

            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                if fig_expr is not None:
                    path_expr = save_dir / f"{self.sample_name}_{gene}_Expression.png"
                    fig_expr.savefig(path_expr, dpi=300, bbox_inches="tight")
                    print(f"{EMOJI['done']} Saved expression plot to {path_expr}")
                if fig_gss is not None:
                    path_gss = save_dir / f"{self.sample_name}_{gene}_GSS.png"
                    fig_gss.savefig(path_gss, dpi=300, bbox_inches="tight")
                    print(f"{EMOJI['done']} Saved GSS plot to {path_gss}")

            if show:
                if fig_expr is not None:
                    fig_expr.show()
                if fig_gss is not None:
                    fig_gss.show()

            return ax_expr, ax_gss

        # Multiple genes: create a single figure with subplots
        import matplotlib.pyplot as plt

        n_panels = len(gene_list) * 2
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes_grid = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes_flat = axes_grid.flatten() if nrows > 1 or ncols > 1 else [axes_grid]

        for idx, g in enumerate(gene_list):
            expr_ax = axes_flat[idx * 2]
            ov.pl.embedding(
                adata_plot,
                basis="spatial",
                color=f"{g}_expr",
                cmap="Reds",
                title=f"{g} Expression",
                size=size,
                show=False,
                ax=expr_ax,
                **kwargs,
            )
            gss_ax = axes_flat[idx * 2 + 1]
            ov.pl.embedding(
                adata_plot,
                basis="spatial",
                color=f"{g}_gss",
                cmap="YlOrRd",
                title=f"{g} GSS",
                size=size,
                show=False,
                ax=gss_ax,
                **kwargs,
            )

        # Hide unused axes
        for idx in range(n_panels, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout()

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            path_out = save_dir / f"{self.sample_name}_multi_gene_Expression_GSS.png"
            fig.savefig(path_out, dpi=300, bbox_inches="tight")
            print(f"{EMOJI['done']} Saved plot to {path_out}")

        if show:
            fig.show()

        return axes_flat[:n_panels]

    def plot_top_genes(
        self,
        trait_name: str,
        top_corr_genes: int = 5,
        save_dir: str | Path | None = None,
        size: float = 50,
        show: bool = True,
        ncols: int = 2,
        **kwargs,
    ):
        """Batch plot expression and GSS for the top-correlated genes.

        Parameters
        ----------
        trait_name : str
            Trait name used to load gene diagnostic info.
        top_corr_genes : int, default 5
            Number of top correlated genes to plot.
        save_dir, size, show, ncols, kwargs
            Forwarded to :meth:`plot_gene_gss`.

        Returns
        -------
        list of matplotlib.axes.Axes
            Axes returned by ``ov.pl.embedding``.
        """
        from ..pp import normalize_total, log1p
        from ..external.gsmap.diagnosis import compute_gene_diagnostic_info, load_gene_diagnostic_info
        from ..external.gsmap._config import DiagnosisConfig

        config = DiagnosisConfig(
            workdir=self.workdir,
            sample_name=self.sample_name,
            annotation=self.annotation or "annotation",
            trait_name=trait_name,
            sumstats_file="",
            top_corr_genes=top_corr_genes,
        )

        adata = self._get_latent_adata()
        if "log1p" not in adata.uns:
            try:
                if adata.X.max() > 14:
                    normalize_total(adata, target_sum=1e4)
                    log1p(adata)
            except Exception:
                pass

        gene_info = load_gene_diagnostic_info(config, adata)
        if gene_info.empty:
            raise ValueError(
                "Gene diagnostic info not found. Please run report() first or provide sumstats_file."
            )

        plot_genes = gene_info.Gene.iloc[:top_corr_genes].tolist()
        return self.plot_gene_gss(
            trait_name=trait_name,
            gene=plot_genes,
            save_dir=save_dir,
            size=size,
            show=show,
            ncols=ncols,
            **kwargs,
        )

    def plot_cauchy_bar(
        self,
        trait_name: str,
        save_path: str | Path | None = None,
        show: bool = True,
        figsize: tuple[float, float] = (6, 4),
        cmap: str = "YlOrRd",
        **kwargs,
    ):
        """Plot a bar chart of -log10(Cauchy p) per annotation.

        Parameters
        ----------
        trait_name : str
            Trait name used to locate the Cauchy-combination result.
        save_path : str or Path, optional
            Path to save the figure.
        show : bool, default True
            Whether to display the figure.
        figsize : tuple, default (6, 4)
            Figure size (width, height).
        cmap : str, default "YlOrRd"
            Matplotlib colormap for gradient bar coloring.
        **kwargs
            Additional arguments forwarded to matplotlib.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        cauchy_path = self._get_cauchy_result_path(trait_name)
        if not cauchy_path.exists():
            raise FileNotFoundError(f"Cauchy result not found: {cauchy_path}")

        cauchy = pd.read_csv(cauchy_path, compression="gzip")
        cauchy["neg_log10_p"] = -np.log10(cauchy["p_cauchy"].clip(lower=1e-300))
        cauchy = cauchy.sort_values("neg_log10_p", ascending=True)

        fig, ax = plt.subplots(figsize=figsize)
        norm = Normalize(vmin=cauchy["neg_log10_p"].min(), vmax=cauchy["neg_log10_p"].max())
        cmap_obj = plt.get_cmap(cmap)
        colors = cmap_obj(norm(cauchy["neg_log10_p"].values))
        bars = ax.barh(cauchy["annotation"], cauchy["neg_log10_p"], color=colors)
        ax.set_xlabel("-log10(Cauchy p)", fontsize=12)
        ax.set_title(f"{trait_name} Cauchy Combination by Annotation", fontsize=12)
        ax.tick_params(axis="both", labelsize=12)
        ax.axvline(x=-np.log10(0.05), color="red", linestyle="--", linewidth=1, label="p=0.05")
        ax.legend(fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"{EMOJI['done']} Saved Cauchy bar plot to {save_path}")
        if show:
            fig.show()
        return fig

    @tracked('run_pipeline', 'ov.genetics.gsmap.run_pipeline',
             adata_attr='adata')
    def run_pipeline(
        self,
        find_latent_kwargs=None,
        latent_to_gene_kwargs=None,
        generate_ldscore_kwargs=None,
        spatial_ldsc_kwargs=None,
        cauchy_combination_kwargs=None,
        report_kwargs=None,
    ) -> dict[str, Path]:
        """Run the currently integrated gsMap pipeline steps in sequence.

        Parameters
        ----------
        find_latent_kwargs : dict, optional
            Keyword arguments for :meth:`find_latent_representation`.
        latent_to_gene_kwargs : dict, optional
            Keyword arguments for :meth:`latent_to_gene`.
        generate_ldscore_kwargs : dict, optional
            Keyword arguments for :meth:`generate_ldscore`.
        spatial_ldsc_kwargs : dict, optional
            Keyword arguments for :meth:`spatial_ldsc`.
        cauchy_combination_kwargs : dict, optional
            Keyword arguments for :meth:`cauchy_combination`.
        report_kwargs : dict, optional
            Keyword arguments for :meth:`report`.

        Returns
        -------
        dict[str, pathlib.Path]
            Dictionary mapping step names to output paths.
        """

        from ..external.gsmap._style import ov_print_done, ov_print_start

        find_latent_kwargs = find_latent_kwargs or {}
        latent_to_gene_kwargs = latent_to_gene_kwargs or {}
        generate_ldscore_kwargs = generate_ldscore_kwargs or {}
        spatial_ldsc_kwargs = spatial_ldsc_kwargs or {}
        cauchy_combination_kwargs = cauchy_combination_kwargs or {}
        report_kwargs = report_kwargs or {}

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{EMOJI['start']} [{ts}] {Colors.CYAN}Running gsMap pipeline in sequence...{Colors.ENDC}"
        )

        latent_h5ad_path = self.find_latent_representation(**find_latent_kwargs)
        marker_score_path = self.latent_to_gene(
            input_hdf5_path=str(latent_h5ad_path),
            **latent_to_gene_kwargs,
        )
        result = {
            "latent_h5ad_path": latent_h5ad_path,
            "marker_score_path": marker_score_path,
        }

        if generate_ldscore_kwargs:
            result["ldscore_dir"] = self.generate_ldscore(**generate_ldscore_kwargs)

        if spatial_ldsc_kwargs:
            result["ldsc_dir"] = self.spatial_ldsc(**spatial_ldsc_kwargs)

        if cauchy_combination_kwargs:
            result["cauchy_file"] = self.cauchy_combination(**cauchy_combination_kwargs)

        if report_kwargs:
            result["report_file"] = self.report(**report_kwargs)

        note(backend='omicverse · gsMap')
        ov_print_done("gsMap pipeline")
        return result


@register_function(
    aliases=["gsmap", "spatial gsmap", "遗传空间映射"],
    category="space",
    description="Create a gsMap runner for genetically informed spatial mapping workflows",
    prerequisites={"functions": [], "optional_functions": []},
    requires={"obsm": ["spatial"]},
    produces={"uns": [], "obs": [], "obsm": []},
    auto_fix="none",
    examples=[
        "runner = ov.space.gsmap(adata, workdir='workdir', sample_name='sample')",
        "config = runner.build_find_latent_config(data_layer='X', n_comps=16)",
        "input_path = runner.prepare_input_h5ad()",
        "latent_path = runner.find_latent_representation(data_layer='X', n_comps=16)",
        "marker_path = runner.latent_to_gene(latent_representation='latent_GVAE')",
        "ldscore_dir = runner.generate_ldscore(gsmap_resource_dir='gsMap_resource')",
        "ldsc_dir = runner.spatial_ldsc(gsmap_resource_dir='gsMap_resource', sumstats_file='trait.sumstats.gz', trait_name='trait')",
        "cauchy_file = runner.cauchy_combination(trait_name='trait', annotation='celltype')",
        "report_file = runner.report(trait_name='trait', sumstats_file='trait.sumstats.gz')",
        "ov.pl.spatial(adata, color=['trait_gsmap_logp'], cmap='YlOrRd', spot_size=1.25)",
        "runner.plot_cauchy_bar(trait_name='trait', cmap='Blues', save_path='cauchy_bar.png')",
        "runner.plot_gene_gss(trait_name='trait', gene='GENE', save_dir='gene_gss')",
        "runner.plot_top_genes(trait_name='trait', top_corr_genes=5, save_dir='top_genes')",
        "runner.plot_manhattan(trait_name='trait', sumstats_file='trait.sumstats.gz', save_path='manhattan.html')",
    ],
    related=["space.svg", "space.pySpaceFlow", "space.GASTON"],
)
def gsmap(adata, *, workdir: str, sample_name: str, annotation: str | None = None) -> _gsmap_runner:
    """Create a lowercase public gsMap runner entry for OmicVerse.

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData object containing at least `adata.obsm["spatial"]`.
    workdir : str
        Output root directory for gsMap intermediate and final files.
    sample_name : str
        Sample identifier used in generated filenames and folders.
    annotation : str, optional
        Optional `adata.obs` annotation column used by gsMap downstream steps.
    """

    return _gsmap_runner(
        adata,
        workdir=workdir,
        sample_name=sample_name,
        annotation=annotation,
    )
