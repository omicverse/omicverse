r"""Time-lapse cell tracking and lineage reconstruction — ``ov.space.track``.

From a live-cell movie (phase contrast, brightfield or fluorescence) to a lineage::

    import omicverse as ov

    movie  = ov.space.track.read_movie('movie.mp4')            # (T, H, W)
    labels = ov.space.track.segment_movie(movie)               # Cellpose cyto3, every frame
    det, tl = ov.space.track.track_cells(labels, movie)        # Trackastra, with divisions
    det, tl = ov.space.track.close_gaps(det, tl, frame_shape=movie.shape[1:])
    sig = ov.space.track.mitosis_signal(movie, cell_diameter=det.diameter.median())
    div = ov.space.track.refine_divisions(det, tl, sig, cell_diameter=det.diameter.median())
    res = ov.space.track.lineage_tables(det, tl, n_frames=len(movie),
                                        frame_shape=movie.shape[1:], divisions=div)

    # or in one call, cached / resumable in workdir (global flow tracker)
    res = ov.space.track.reconstruct_lineage('movie.mp4', workdir='lineage_work')

    # global tracker on an existing label movie
    res = ov.space.track.track_lineage(labels, movie)   # build_tracklets -> flow_track

``res`` holds ``cells`` (per tracklet: start, end, parent, generation, founder,
fate), ``divisions`` (frame, x, y, generation of the mother, founder), ``founders``
(position in the first frame, divisions in the lineage), ``generation_outcome`` and
``generation_window`` count matrices.

``detect_mitoses`` finds mitoses directly from the pixels (no tracking) and is a
useful independent check of how many divisions the tree should contain.

Conventions: frame indices are those of the decoded movie; ``x`` to the right,
``y`` downwards, origin at the top-left pixel; a cell whose birth is not seen
(present in frame 0 or entering across the border) is generation 0; a daughter is
mother + 1; a division's frame is the last frame of the single mother.

``track_lineage`` / ``flow_track`` assemble overlap tracklets with one integer
program (flow conservation: 0 / 1 / 2 cells per mask, gap closing, divisions weighted
by the mitotic cue, a minimum cell cycle measured on the data). They hold the
lineage together where frame-to-frame linkers break it (missed detections, touching
cells, daughters that stay in contact).

Methods: Cellpose (Stringer et al., Nat. Methods 2021; cyto3: 2025), Trackastra
(Gallusser & Weigert, ECCV 2024), conservation tracking (Schiegg et al., ICCV 2013). GPU strongly recommended for long movies.
"""
from ._movie import read_movie, segment_movie
from ._track import detections_table, track_cells, close_gaps, close_merge_gaps
from ._mitosis import mitosis_signal, detect_mitoses, refine_divisions, prune_divisions, insert_mitoses, adopt_orphans, consensus_divisions
from ._lineage import lineage_tables, lineage_json, OUTCOMES
from ._pipeline import reconstruct_lineage, repair_tree
from ._flow import (frame_overlaps, build_tracklets, tracklet_table, flow_track,
                    flow_lineage_tables, track_lineage)

__all__ = ["read_movie", "segment_movie", "detections_table", "track_cells", "close_gaps", "close_merge_gaps",
           "mitosis_signal", "detect_mitoses", "refine_divisions", "prune_divisions", "insert_mitoses", "adopt_orphans", "consensus_divisions", "lineage_tables",
           "reconstruct_lineage", "repair_tree", "lineage_json", "OUTCOMES",
           "frame_overlaps", "build_tracklets", "tracklet_table", "flow_track",
           "flow_lineage_tables", "track_lineage"]
