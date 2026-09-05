# Spatial count-likelihood inputs

cell2location and RCTD require raw integer-like counts. Preserve them in a layer:

```python
decov.deconvolution(
    method='cell2location', celltype_key_sc='cell_type',
    counts_layer_sc='counts', counts_layer_sp='counts',
)
```

Custom layer names are supported. A missing custom layer is an error. When the
default counts layer is absent, X is accepted only when all values are finite,
non-negative and integer-like. Continuous normalized data is never rounded.
Reference gene filtering uses counts temporarily and restores the original X;
both cell2location likelihood registries explicitly use their count layer.
RCTD likewise receives count-matrix copies.

Starfysh Visium image patches use x=imagecol, y=imagerow. Multi-library Starfysh
input is rejected instead of silently selecting the first image. Unsupported
platforms and missing signatures fail before heavy backend work.

Validation uses real one-epoch cell2location reference/spatial training and a
real RCTD full-mode run in an isolated environment, plus mocked routing tests.
The RCTD runtime fixture has >300 UMI per spot, as required by its sigma
estimator. Its approximately 403 MB Q-matrix asset was downloaded to the test
environment on D:, not the system disk. No CI or package configuration changed.
