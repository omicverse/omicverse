# Spatial model and runtime corrections

Recompute SpaceFlow embeddings affected by the duplicated spatial sampling
index or aliased best-state snapshots. Recompute pseudospatial roots that were
stored as subsample positions rather than global observation indices.

STAGATE rejects zero epochs and prediction before training. Its constructor
resolves explicit coordinate keys on a working copy; the original spatial
embedding remains unchanged. When both legacy obs X/Y and obsm['spatial'] are
present, an omitted key preserves legacy precedence with a migration warning.
Pass spatial_key='spatial' to choose canonical coordinates explicitly.

clusters() rejects unknown method names, copies caller configuration before
merging defaults, and accepts dense CAST inputs. merge_cluster() maps actual
category names, including nonnumeric labels. STT accepts canonical spatial
coordinates while warning when legacy xy_loc is selected implicitly.

Tangram constant projected columns remain finite. Geometry errors identify
the actual POT dependency. The nonexistent H&E BLEEP backend is no longer
advertised. These changes do not claim full runtime validation of every heavy
backend, particularly H&E models. No new dependency tier, CI, or setup.py edit.

Local checks: 42 focused tests passed, including real tiny SpaceFlow, STAGATE,
Tangram and POT geometry runs plus dispatcher/coordinate/selection contracts.
