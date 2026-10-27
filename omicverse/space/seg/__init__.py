r"""Cell segmentation for spatial transcriptomics — ``ov.space.seg``.

One entry point per input type, so methods can be swapped and compared on the
same sample, as in the Nature Methods 2026 benchmark of segmentation methods
for spatial transcriptomics (Martinez-de-Morentin et al.)::

    import omicverse as ov

    ov.space.seg.methods()                                   # what is installed

    # image-based: DAPI / morphology / H&E, any size (tiled, seam-free)
    nuclei = ov.space.seg.segment(dapi, method='cellpose', diameter=30, gpu=True)
    nuclei = ov.space.seg.segment(he_rgb, method='stardist', image_type='he')
    masks  = ov.space.seg.segment(dapi, method='sam2')       # also sam, micro_sam, cellsam
    cells  = ov.space.seg.expand_labels(nuclei, distance=10)  # nucleus -> cell

    # molecules -> cells -> AnnData (Xenium / CosMx / MERFISH / Stereo-seq GEM)
    tx['cell'] = ov.space.seg.assign_transcripts(tx, cells, pixel_size=0.2125)
    adata = ov.space.seg.aggregate(tx, cell='cell', gene='feature_name',
                                   labels=cells, pixel_size=0.2125)
    ov.pl.spatialseg(adata, color='n_transcripts')

    # transcript-based: Baysor, optionally with the image segmentation as prior
    tx['nucleus'] = ov.space.seg.assign_transcripts(tx, nuclei, pixel_size=0.2125)
    tx = ov.space.seg.segment_transcripts(tx, method='baysor', gene='feature_name',
                                          prior='nucleus', prior_confidence=0.5)

    # reference-free CellSPA metrics and a ranked comparison
    table = ov.space.seg.compare({
        'cellpose': ov.space.seg.evaluate(cells, tx),
        'baysor':   ov.space.seg.evaluate(None, tx_baysor),
    })

Methods
-------
``watershed``  Spateo-style marker watershed; scikit-image only.
``cellpose``   Cellpose 2/3/4; ``restore_type='denoise_cyto3'`` (v3) and
               Cellpose-SAM ``'cpsam'`` (v4, default when installed).
``stardist``   StarDist ``2D_versatile_fluo`` / ``2D_versatile_he``.
``sam``        Segment Anything ViT-H automatic masks (benchmark settings).
``sam2``       SAM 2 ``facebook/sam2-hiera-large`` automatic masks.
``micro_sam``  micro-SAM ``vit_b_lm`` (instance decoder, or ``amg=True``).
``cellsam``    CellSAM foundation model.
``baysor``     Transcript-based MRF segmentation (CLI), image-free or with prior.

Visium HD uses the same backends through
``ov.space.visium_10x_hd_cellpose_he(adata, method='stardist')`` and
``ov.space.visium_10x_hd_cellpose_gex``.

Install
-------
``pip install 'omicverse[seg]'`` adds Cellpose, Segment Anything and SAM 2.
StarDist (``pip install stardist tensorflow``), micro-SAM (conda-forge),
CellSAM (GitHub) and Baysor (binary) are installed separately;
:func:`methods` reports what is available.
"""
from ._api import expand_labels, load_image, methods, segment
from ._metrics import compare, evaluate
from ._tiling import segment_tiled
from ._transcripts import aggregate, assign_transcripts, segment_transcripts

__all__ = [
    "methods",
    "segment",
    "segment_tiled",
    "expand_labels",
    "load_image",
    "assign_transcripts",
    "aggregate",
    "segment_transcripts",
    "evaluate",
    "compare",
]
