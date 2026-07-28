r"""mRNA therapeutics design — joint codon + structure optimisation.

Designing an mRNA (vaccine / protein-replacement therapeutic) is not just codon
optimisation: the *secondary structure* controls stability and translation, and
the two objectives trade off. :func:`mrna_design` exposes both a transparent
baseline and the real **LinearDesign** algorithm (Zhang *et al.*, *Nature* 2023)
— a lattice/CKY dynamic program that finds, in linear time, the mRNA of minimum
folding free energy (most stable) subject to a codon-optimality (CAI) reward,
tuned by ``lambda``:

* ``method='baseline'`` — codon-optimise (DNAchisel) then report the resulting
  MFE and CAI. Fast, dependency-light.
* ``method='lineardesign'`` — the real LinearDesign C++ solver, auto-cloned and
  compiled on first use. Returns the jointly MFE/CAI-optimal mRNA.

``lambda`` balances the two objectives (0 = pure MFE / most-structured;
higher = more codon-optimised / higher CAI).
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from .._registry import register_function
from ._esm_common import weights_dir

_REPO_URL = "https://github.com/LinearDesignSoftware/LinearDesign"
_HOSTS = {"human": "codon_usage_freq_table_human.csv",
          "yeast": "codon_usage_freq_table_yeast.csv"}


@dataclass
class MRNADesign:
    protein: str
    mrna: str                # designed mRNA (RNA, 5'->3')
    structure: str           # dot-bracket MFE structure
    mfe: float               # folding free energy (kcal/mol; more negative = stabler)
    #: CAI against the requested host, computed by :func:`~omicverse.synbio.cai`
    #: for every method — so designs from different backends are comparable.
    cai: float
    method: str
    lam: float = 0.0
    #: What the backend reported for itself, on its own codon table. Kept because
    #: it is what the tool printed, but it is *not* comparable across methods.
    cai_reported_by_tool: Optional[float] = None

    def __repr__(self) -> str:  # pragma: no cover
        return (f"MRNADesign(len={len(self.mrna)}nt, MFE={self.mfe:.1f} kcal/mol, "
                f"CAI={self.cai:.3f}, {self.method})")


# ---------------------------------------------------------------------------
# vendored LinearDesign
# ---------------------------------------------------------------------------
def _libdir(repo: str) -> str:
    return os.path.join(repo, "src", "Utils", "libraries")


def ensure_lineardesign() -> str:
    """Clone + compile LinearDesign on first use; return the repo path.

    The bundled energy library ships as an old-C++-ABI ``.so`` linked against an
    ancient glibc, so we compile ``linear_design.cpp`` with
    ``-D_GLIBCXX_USE_CXX11_ABI=0`` against ``LinearDesign_linux64_old.so`` (the
    modern ``.so`` needs GLIBC_2.29, absent on many HPC images)."""
    repo = os.path.join(weights_dir(), "LinearDesign")
    binary = os.path.join(repo, "bin", "LinearDesign_2D")
    if os.path.exists(binary):
        return repo
    if not os.path.exists(os.path.join(repo, "src", "linear_design.cpp")):
        try:
            subprocess.run(["git", "clone", "--depth", "1", _REPO_URL, repo],
                           check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
            raise ImportError(
                "ov.synbio.mrna_design(method='lineardesign') 需要 LinearDesign,"
                f"自动 git clone 失败。请手动 git clone {_REPO_URL} 到 {repo}。"
                f"({getattr(exc, 'stderr', exc)})") from exc
    os.makedirs(os.path.join(repo, "bin"), exist_ok=True)
    old_so = os.path.join(_libdir(repo), "LinearDesign_linux64_old.so")
    cxx = os.environ.get("CXX", "g++")
    cmd = [cxx, "-std=c++11", "-O2", "-DFINAL_CHECK", "-DSPECIAL_HP",
           "-fpermissive", "-D_GLIBCXX_USE_CXX11_ABI=0",
           os.path.join(repo, "src", "linear_design.cpp"),
           "-o", binary, old_so]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(binary):
        raise ImportError(
            "LinearDesign 编译失败(需要 C++11 编译器,如 `ml load devel gcc`)。"
            f"\n{proc.stderr[-800:]}")
    return repo


def _run_lineardesign(protein: str, lam: float, host: str) -> MRNADesign:
    repo = ensure_lineardesign()
    binary = os.path.join(repo, "bin", "LinearDesign_2D")
    codon_csv = _HOSTS.get(host, _HOSTS["human"])
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [_libdir(repo), env.get("LD_LIBRARY_PATH", "")])
    proc = subprocess.run(
        [binary, str(float(lam)), "0", codon_csv],
        input=protein.strip().upper() + "\n", capture_output=True, text=True,
        cwd=repo, env=env)
    out = proc.stdout
    mrna = struct = ""
    mfe = cai = 0.0
    for line in out.splitlines():
        line = line.strip()
        if "mRNA sequence:" in line:
            mrna = line.split("mRNA sequence:")[1].strip()
        elif "mRNA structure:" in line:
            struct = line.split("mRNA structure:")[1].strip()
        elif "free energy:" in line:
            # "mRNA folding free energy: -6.00 kcal/mol; mRNA CAI: 0.910"
            try:
                mfe = float(line.split("free energy:")[1].split("kcal")[0])
                if "CAI:" in line:
                    cai = float(line.split("CAI:")[1].strip())
            except (ValueError, IndexError):
                pass
    if not mrna:
        raise RuntimeError("LinearDesign 未返回 mRNA:\n" + (proc.stderr or out)[-600:])
    # Recompute CAI with *our* implementation and the host we were asked for, so
    # that a baseline design and a LinearDesign design are on one yardstick.
    # LinearDesign's own CAI comes from its own codon table; reporting the two
    # side by side as if they were the same quantity made the baseline look worse
    # on CAI than it is.
    return MRNADesign(protein=protein.upper(), mrna=mrna, structure=struct,
                      mfe=mfe, cai=_cai_of(mrna, host),
                      cai_reported_by_tool=cai,
                      method="lineardesign", lam=lam)


#: ``host`` names accepted by :func:`mrna_design` mapped to codon-table names.
_CAI_HOSTS = {"human": "h_sapiens", "yeast": "s_cerevisiae"}


def _cai_of(sequence: str, host: str) -> float:
    """CAI of a designed sequence against the host it was designed for.

    ``cai()`` defaults to ``host='e_coli'``. Calling it without ``host=`` on a
    human-optimised CDS scored the design against the wrong organism entirely:
    a human design read 0.656 that way and 0.990 against the human table, which
    inverted the baseline-versus-LinearDesign comparison.
    """
    import warnings

    from ._expression import cai as _cai
    dna = str(sequence).upper().replace("U", "T")
    try:
        return float(_cai(dna, host=_CAI_HOSTS.get(host, "h_sapiens")))
    except Exception as exc:
        warnings.warn(
            f"CAI 没算出来({exc});返回 nan 而不是 0.0 —— 0.0 会被读成"
            f"'密码子适配极差',而实际情况是没算成。", stacklevel=2)
        return float("nan")


def _baseline_design(protein: str, host: str) -> MRNADesign:
    from ._codon import codon_optimize
    from ._rna import rna_fold
    cds = codon_optimize(protein, host=_CAI_HOSTS.get(host, "h_sapiens")).sequence
    fold = rna_fold(cds)
    return MRNADesign(protein=protein.upper(), mrna=cds.replace("T", "U"),
                      structure=fold.structure, mfe=fold.mfe,
                      cai=_cai_of(cds, host), method="baseline")


@register_function(
    aliases=["mrna_design", "mRNA设计", "mrna_optimization", "mRNA优化",
             "疫苗mRNA设计", "lineardesign", "信使RNA设计", "mrna_optimize"],
    category="synthetic_biology",
    description="mRNA 治疗/疫苗设计:密码子+二级结构联合优化。method='lineardesign' 走真实 LinearDesign(Zhang 2023 Nature,线性时间联合最优 MFE/CAI,自动克隆编译);method='baseline' 用 DNAchisel 密码子优化后报告 MFE/CAI。lambda 平衡稳定性(MFE)与密码子适应(CAI)。Joint codon+structure mRNA design (LinearDesign or baseline).",
    examples=[
        "d = ov.synbio.mrna_design('MNDTEAI', method='lineardesign', lam=3)",
        "d.mrna, d.mfe, d.cai",
    ],
    related=["synbio.codon_optimize", "synbio.rna_fold", "synbio.utr5_strength"],
    requires={},
    produces={},
)
def mrna_design(protein: str, method: str = "lineardesign", lam: float = 0.0,
                host: str = "human") -> MRNADesign:
    """Design an mRNA coding for *protein* (single-letter amino acids).

    ``method='lineardesign'`` runs the real LinearDesign solver (jointly optimal
    MFE + CAI, tuned by ``lam``); ``method='baseline'`` codon-optimises with
    DNAchisel and reports the resulting structure/CAI. ``host`` selects the codon
    usage table (``'human'`` or ``'yeast'``)."""
    if method not in ("lineardesign", "baseline"):
        raise ValueError(
            f"method must be one of ['lineardesign', 'baseline'], got {method!r}")
    if host not in _HOSTS:
        raise ValueError(f"host must be one of {list(_HOSTS)}, got {host!r}")
    prot = "".join(protein.split()).upper()
    if method == "baseline":
        return _baseline_design(prot, host)
    return _run_lineardesign(prot, lam, host)


__all__ = ["mrna_design", "MRNADesign", "ensure_lineardesign"]
