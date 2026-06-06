"""Generate ``centromere_<genome>.csv`` for ``ov.pl.cnv_heatmap(split_arms=True)``.

General fetch-and-derive utility (not imported at runtime — the plot code only
reads the produced CSVs). Run manually to (re)generate or add a genome::

    python generate_centromeres.py hg38 hg19 mm10 mm39

For each genome it downloads the UCSC ``cytoBand`` table (falling back to
``cytoBandIdeo``), takes the boundary between the two ``acen`` (centromere)
cytobands per chromosome, and writes ``chromosome,centromere`` rows for the
standard autosomes + X/Y.

Genomes whose UCSC cytoBand carries **no** ``acen`` bands (telocentric mouse
assemblies such as mm10) are covered by ``_HARDCODED`` instead — mouse
chromosomes are acrocentric/telocentric, so the centromere sits at the proximal
~3 Mb and there is effectively no p arm.
"""
from __future__ import annotations

import csv
import gzip
import io
import re
import sys
import urllib.request
from pathlib import Path

_UCSC = "http://hgdownload.cse.ucsc.edu/goldenPath/{g}/database/{f}.txt.gz"
_STD = re.compile(r"^chr(\d+|X|Y)$")

# Genomes UCSC cytoBand cannot supply an acen-derived centromere for.
# mm10: mouse chromosomes are telocentric — centromere at the proximal end;
# UCSC mm10 cytoBand has no acen bands. ~3 Mb pericentromeric boundary
# (approximate; mouse has essentially no p arm). Override if you have exact
# values.
_HARDCODED: dict[str, dict[str, int]] = {
    "mm10": {f"chr{c}": 3_000_000 for c in [*range(1, 20), "X", "Y"]},
}


def fetch_cytoband(genome: str):
    """Return list of (chrom, start, end, name, stain); prefer a table with acen."""
    last = None
    for fname in ("cytoBand", "cytoBandIdeo"):
        try:
            raw = urllib.request.urlopen(_UCSC.format(g=genome, f=fname), timeout=60).read()
            rows = []
            with gzip.open(io.BytesIO(raw), "rt") as fh:
                for line in fh:
                    chrom, start, end, name, stain = line.rstrip("\n").split("\t")
                    rows.append((chrom, int(start), int(end), name, stain))
            if any(r[4] == "acen" for r in rows):
                return rows
            last = rows
        except Exception as exc:  # noqa: BLE001
            last = exc
    return last


def derive_centromeres(rows) -> dict[str, int]:
    """Centromere bp per chromosome = p-arm acen end (== q-arm acen start)."""
    by_chrom: dict[str, list] = {}
    for chrom, start, end, name, stain in rows:
        if stain == "acen" and _STD.match(chrom):
            by_chrom.setdefault(chrom, []).append((start, end, name))
    out: dict[str, int] = {}
    for chrom, bands in by_chrom.items():
        p_ends = [e for _s, e, n in bands if n.startswith("p")]
        q_starts = [s for s, _e, n in bands if n.startswith("q")]
        out[chrom] = max(p_ends) if p_ends else min(q_starts)
    return out


def _chrom_key(chrom: str):
    s = chrom[3:]
    return (0, int(s)) if s.isdigit() else (1, s)


def generate(genome: str, out_dir: str | Path) -> Path:
    """Fetch/derive and write centromere_<genome>.csv; return the path."""
    if genome in _HARDCODED:
        cen = dict(_HARDCODED[genome])
    else:
        rows = fetch_cytoband(genome)
        if not isinstance(rows, list):
            raise RuntimeError(f"could not fetch UCSC cytoBand for {genome!r}: {rows}")
        cen = derive_centromeres(rows)
        if not cen:
            raise RuntimeError(
                f"{genome!r}: UCSC cytoBand has no 'acen' bands — add it to _HARDCODED."
            )
    path = Path(out_dir) / f"centromere_{genome}.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["chromosome", "centromere"])
        for chrom, pos in sorted(cen.items(), key=lambda kv: _chrom_key(kv[0])):
            writer.writerow([chrom, int(pos)])
    return path


if __name__ == "__main__":
    here = Path(__file__).parent
    for g in sys.argv[1:] or ["hg38", "hg19", "mm10"]:
        print("wrote", generate(g, here))
