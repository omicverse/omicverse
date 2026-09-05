# Spatial coordinate correctness

The public coordinate convention is x=image column, y=image row in full-resolution
pixels for Visium. Tests compare real 10x HDF5 matrices and legacy CSV, modern
CSV and parquet position files against Scanpy's Visium output. Standard modern
files retain the same coordinates; the old rename/select sequence could cancel
its axis-name swap. This is not a claim that all old modern outputs were reversed.
The legacy no-header reader now retains the first barcode. Named row/column
metadata is now correct and used directly rather than positionally renamed.

Use crop_space_visium(..., coordinate_order='xy') for (x,y)/(width,height).
Omitting coordinate_order temporarily retains legacy yx with a warning. Manual
translation's offset_mode='absolute' applies positive x/right, y/down; omitted
mode preserves the old subtraction behavior with a migration warning.

For multi-library objects, choose both library_id and library_key explicitly.
Rotation/manual translation only changes the chosen library. Crop returns the
selected region/library. Crop and rotation transform every interpretable image
resolution using its own scale factor. Images without a usable mapping are
removed only from the output copy, with serializable removal metadata; input
data remains intact. Fractional lowres crop origins use aligned interpolation.

map_spatial_auto remains an experimental image-correlation heuristic; this PR
fixes library selection, hard-coded columns and temporary files, not full affine
registration accuracy. scale != 1 crop remains explicitly unsupported.

Local tests: 20 tool contracts plus 5 reader tests pass, including non-symmetric
coordinates, library isolation, multi-resolution landmarks and H5AD roundtrip.
No CI, setup.py or dependency metadata changes.
