# Vendored centromere positions

`centromere_<genome>.csv` — per-chromosome centromere coordinate (1-based bp)
used by `ov.pl.cnv_heatmap(split_arms=True)` to draw the p/q arm boundary.
Bundled: `hg38`, `hg19`, `mm10`.

A gene/bin with `start < centromere` is assigned to the **p** arm, otherwise the
**q** arm. Standard autosomes + chrX/chrY only.

## Regenerate / add a genome
Use the bundled generator — input a genome name, it retrieves UCSC and writes
the CSV (no manual hardcoding for human builds):

```bash
python generate_centromeres.py hg38 hg19 mm10 mm39
```

It downloads the UCSC `cytoBand` table (`goldenPath/<genome>/database/cytoBand.txt.gz`,
falling back to `cytoBandIdeo`) and takes the boundary between the two
`acen`-stained cytobands per chromosome (p-arm `acen` end == q-arm `acen` start).

## Caveats
- **Source**: UCSC cytoBand, fetched 2026-06-06 (hg38/hg19 acen-derived).
- **mm10 is hardcoded** in the generator (`_HARDCODED`): mouse chromosomes are
  telocentric/acrocentric and UCSC mm10 `cytoBand` carries **no `acen` bands**,
  so there is effectively no p arm — centromere is set to the proximal ~3 Mb as
  an approximation. Arm-splitting mouse data is therefore of limited value
  (most genes fall on the q arm). Newer mouse assembly `mm39` *does* have `acen`
  in `cytoBandIdeo` and can be generated normally.
