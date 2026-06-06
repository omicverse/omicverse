# Vendored centromere positions

`centromere_hg38.csv` / `centromere_hg19.csv` — per-chromosome centromere
coordinate (1-based bp) used by `ov.pl.cnv_heatmap(split_arms=True)` to draw the
p/q arm boundary.

## Source & derivation
- Source: UCSC Genome Browser `cytoBand` table
  (`goldenPath/<build>/database/cytoBand.txt.gz`), downloaded 2026-06-06.
- The centromere coordinate is the boundary between the two `acen`-stained
  cytobands of each chromosome (i.e. the end of the p-arm `acen` band ==
  the start of the q-arm `acen` band).
- A gene/bin with `start < centromere` is assigned to the **p** arm, otherwise
  to the **q** arm.
- Standard autosomes + chrX/chrY only.

To regenerate: download `cytoBand.txt.gz` for the build, keep rows with
`gieStain == "acen"`, and for each chromosome take `max(end)` over the
`p*`-named acen band(s).
