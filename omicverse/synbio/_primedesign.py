r"""Vendored `PrimeDesign <https://github.com/pinellolab/PrimeDesign>`_ — the
Pinello-lab reference pegRNA/ngRNA designer (Hsu *et al.* 2021).

PrimeDesign is a pure-stdlib Python tool (no heavy deps). We clone it on first
use and drive its command-line script, converting an ``ov.synbio`` edit
(``target`` + ``edit_pos`` + ``ref`` + ``alt``) into PrimeDesign's inline
bracket notation — substitution ``(G/A)``, insertion ``(+ATCG)``, deletion
``(-ATCG)`` — and parsing the returned pegRNA/ngRNA table back into
:class:`~omicverse.synbio._editing.PegRNA` objects.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from typing import List, Optional

from ._esm_common import weights_dir

_REPO_URL = "https://github.com/pinellolab/PrimeDesign"


def _script() -> str:
    repo = os.path.join(weights_dir(), "PrimeDesign")
    return os.path.join(repo, "PrimeDesign", "command_line", "primedesign.py")


def ensure_primedesign() -> str:
    """Clone PrimeDesign on first use; return the command-line script path."""
    script = _script()
    if os.path.exists(script):
        return script
    repo = os.path.join(weights_dir(), "PrimeDesign")
    try:
        subprocess.run(["git", "clone", "--depth", "1", _REPO_URL, repo],
                       check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
        raise ImportError(
            "prime_editing_design(method='primedesign') 需要 PrimeDesign,"
            f"自动 git clone 失败。请手动 git clone {_REPO_URL} 到 {repo}。"
            f"({getattr(exc, 'stderr', exc)})") from exc
    if not os.path.exists(script):  # pragma: no cover
        raise ImportError(f"PrimeDesign 脚本缺失:{script}")
    return script


def _edit_notation(target: str, edit_pos: int, ref: str, alt: str) -> str:
    """Insert PrimeDesign's bracket edit annotation at *edit_pos*."""
    ref, alt = ref.upper(), alt.upper()
    if ref and alt:
        note = f"({ref}/{alt})"          # substitution
    elif alt and not ref:
        note = f"(+{alt})"               # insertion
    elif ref and not alt:
        note = f"(-{ref})"               # deletion
    else:
        raise ValueError("必须指定 ref/alt 之一(替换/插入/缺失)。")
    return target[:edit_pos] + note + target[edit_pos + len(ref):]


def run_primedesign(target: str, edit_pos: int, ref: str, alt: str,
                    n_pegrnas: int = 5, n_ngrnas: int = 3,
                    pbs_list: Optional[List[int]] = None,
                    rtt_list: Optional[List[int]] = None) -> List["PegRNA"]:
    """Run PrimeDesign for one edit and return ranked :class:`PegRNA` objects
    (each pegRNA carries its best ngRNA as the PE3 nick)."""
    from ._editing import PegRNA
    script = ensure_primedesign()
    seq = _edit_notation(target.upper().replace("U", "T"), edit_pos, ref, alt)

    work = tempfile.mkdtemp(prefix="ovsynbio_primedesign_")
    infile = os.path.join(work, "input.csv")
    with open(infile, "w") as fh:
        fh.write("target_name,target_sequence\nedit,%s\n" % seq)

    cmd = [sys.executable, script, "-f", infile, "-out", work,
           "-n_pegrnas", str(n_pegrnas), "-n_ngrnas", str(n_ngrnas)]
    if pbs_list:
        cmd += ["-pbs", *map(str, pbs_list)]
    if rtt_list:
        cmd += ["-rtt", *map(str, rtt_list)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=work)

    out_csv = None
    for f in os.listdir(work):
        if f.endswith("_PrimeDesign.csv"):
            out_csv = os.path.join(work, f)
            break
    if out_csv is None:
        raise RuntimeError(
            "PrimeDesign 未产出结果:\n" + (proc.stderr or proc.stdout)[-1000:])

    # group rows by pegRNA number: one pegRNA row + its ngRNA rows
    pegs: dict = {}
    ngrnas: dict = {}
    with open(out_csv) as fh:
        for row in csv.DictReader(fh):
            num = row.get("pegRNA_number", "")
            gtype = (row.get("gRNA_type") or "").lower()
            if gtype == "pegrna":
                pegs[num] = row
            elif gtype == "ngrna":
                ngrnas.setdefault(num, []).append(row)

    out: List[PegRNA] = []
    for num, row in pegs.items():
        ext = (row.get("Extension_sequence") or "").upper()
        pbs_len = int(float(row.get("PBS_length") or 0))
        rtt_len = int(float(row.get("RTT_length") or 0))
        rtt = ext[:rtt_len] if rtt_len else ext
        pbs = ext[rtt_len:] if rtt_len else ""
        ng = ngrnas.get(num, [])
        pe3 = None
        if ng:
            g = ng[0]
            pe3 = {"spacer": (g.get("Spacer_sequence") or "").upper(),
                   "nick_position": int(float(g.get("Nick_index") or 0)),
                   "distance": int(float(g.get("ngRNA-to-pegRNA_distance") or 0)),
                   "annotation": g.get("Annotation", "")}
        out.append(PegRNA(
            spacer=(row.get("Spacer_sequence") or "").upper(),
            pbs=pbs, rtt=rtt, extension_3p=ext,
            strand="+" if (row.get("Strand") or "+") == "+" else "-",
            pam=(row.get("PAM_sequence") or "").upper(),
            nick_position=int(float(row.get("Nick_index") or 0)),
            nick_to_edit=int(float(row.get("pegRNA-to-edit_distance") or 0)),
            edit=f"{ref or '-'}>{alt or '-'}@{edit_pos}", pe3_nick=pe3))
    out.sort(key=lambda p: p.nick_to_edit)
    return out
