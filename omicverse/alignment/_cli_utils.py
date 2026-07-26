"""Shared CLI helpers for alignment wrappers."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, TypeVar, cast


_INSTALL_HINTS = {
    "prefetch": "conda install -c bioconda -y sra-tools",
    "vdb-validate": "conda install -c bioconda -y sra-tools",
    "fasterq-dump": "conda install -c bioconda -y sra-tools",
    "fastp": "conda install -c bioconda -y fastp",
    "STAR": "conda install -c bioconda -y star",
    "samtools": "conda install -c bioconda -y samtools",
    "featureCounts": "conda install -c bioconda -y subread",
    "pigz": "conda install -c conda-forge -y pigz",
    "gzip": "conda install -c conda-forge -y gzip",
    "vsearch": "conda install -c bioconda -y vsearch",
    "cutadapt": "pip install cutadapt",
    "diamond": "conda install -c bioconda -y diamond",
    "blastp": "conda install -c bioconda -y blast",
    "blastn": "conda install -c bioconda -y blast",
    "blastx": "conda install -c bioconda -y blast",
    "tblastn": "conda install -c bioconda -y blast",
    "tblastx": "conda install -c bioconda -y blast",
    "makeblastdb": "conda install -c bioconda -y blast",
    "tracy": "conda install -c bioconda -y tracy",
    "emu": "pip install emu",
    "Rscript": "conda install -c conda-forge -y r-base bioconductor-dada2",
    "mafft": "conda install -c bioconda -y mafft",
    "FastTree": "conda install -c bioconda -y fasttree",
    "FastTreeMP": "conda install -c bioconda -y fasttree",
}


_INSTALL_COMMANDS = {
    "prefetch": ["sra-tools", "-c", "bioconda"],
    "vdb-validate": ["sra-tools", "-c", "bioconda"],
    "fasterq-dump": ["sra-tools", "-c", "bioconda"],
    "fastp": ["fastp", "-c", "bioconda"],
    "STAR": ["star", "-c", "bioconda"],
    "samtools": ["samtools", "-c", "bioconda"],
    "featureCounts": ["subread", "-c", "bioconda"],
    "pigz": ["pigz", "-c", "conda-forge"],
    "gzip": ["gzip", "-c", "conda-forge"],
    "vsearch": ["vsearch", "-c", "bioconda"],
    "cutadapt": ["cutadapt", "-c", "bioconda"],
    "mafft": ["mafft", "-c", "bioconda"],
    "FastTree": ["fasttree", "-c", "bioconda"],
    "FastTreeMP": ["fasttree", "-c", "bioconda"],
    # `_INSTALL_HINTS` is only the message shown when a tool is missing; this is
    # the dict `_install_tool` actually dispatches on. A tool listed in one but
    # not the other makes `auto_install=True` a silent no-op that still ends in
    # FileNotFoundError, so the two must be kept in step.
    "diamond": ["diamond", "-c", "bioconda"],
    "blastp": ["blast", "-c", "bioconda"],
    "blastn": ["blast", "-c", "bioconda"],
    "blastx": ["blast", "-c", "bioconda"],
    "tblastn": ["blast", "-c", "bioconda"],
    "tblastx": ["blast", "-c", "bioconda"],
    "makeblastdb": ["blast", "-c", "bioconda"],
    "tracy": ["tracy", "-c", "bioconda"],
}




def _env_prefix() -> Optional[str]:
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        return prefix
    if sys.prefix:
        return sys.prefix
    return None



def _root_prefix_from_envs_path() -> Optional[str]:
    prefix = Path(sys.prefix).resolve()
    for parent in prefix.parents:
        if parent.name == "envs":
            return str(parent.parent)
    return None

def _root_prefix_from_state() -> Optional[str]:
    state_path = Path(sys.prefix) / "conda-meta" / "state"
    if not state_path.exists():
        return None
    try:
        import json
        data = json.loads(state_path.read_text())
        root = data.get("root_prefix")
        if root:
            return str(Path(root).expanduser())
    except Exception:
        return None
    return None

def _available_installer() -> Optional[str]:
    env_candidates = [
        os.environ.get("OMICVERSE_MAMBA_EXE"),
        os.environ.get("OMICVERSE_CONDA_EXE"),
        os.environ.get("OMICVERSE_MICROMAMBA_EXE"),
        os.environ.get("MAMBA_EXE"),
        os.environ.get("CONDA_EXE"),
        os.environ.get("MICROMAMBA_EXE"),
    ]
    for cand in env_candidates:
        if cand and Path(cand).exists():
            return cand

    for name in ("mamba", "conda", "micromamba"):
        if shutil.which(name):
            return name

    root_candidates = [
        os.environ.get("MAMBA_ROOT_PREFIX"),
        os.environ.get("CONDA_ROOT_PREFIX"),
        _root_prefix_from_envs_path(),
        _root_prefix_from_state(),
    ]
    for root in root_candidates:
        if root:
            root_path = Path(root).expanduser().resolve()
            for subdir in ("condabin", "bin", "Scripts"):
                for exe in ("mamba", "conda", "micromamba", "mamba.exe", "conda.exe", "micromamba.exe"):
                    cand = root_path / subdir / exe
                    if cand.exists() and os.access(cand, os.X_OK):
                        return str(cand)

    prefix = Path(sys.prefix).resolve()
    search_roots = [prefix] + list(prefix.parents[:4])
    bin_dirs = ("condabin", "bin", "Scripts")
    exe_names = ("mamba", "conda", "micromamba", "mamba.exe", "conda.exe", "micromamba.exe")
    for root in search_roots:
        for subdir in bin_dirs:
            base = root / subdir
            for exe in exe_names:
                cand = base / exe
                if cand.exists() and os.access(cand, os.X_OK):
                    return str(cand)

    common_roots = [
        "~/.conda",
        "~/miniconda3",
        "~/miniforge3",
        "~/mambaforge",
        "~/anaconda3",
        "/opt/conda",
        "/opt/miniconda",
        "/opt/miniconda3",
        "/opt/miniforge3",
        "/opt/mambaforge",
        "/usr/local/miniconda3",
        "/usr/local/anaconda3",
    ]
    for root in common_roots:
        root_path = Path(root).expanduser()
        for subdir in bin_dirs:
            for exe in exe_names:
                cand = root_path / subdir / exe
                if cand.exists() and os.access(cand, os.X_OK):
                    return str(cand)

    return None


def _install_tool(name: str) -> bool:
    spec = _INSTALL_COMMANDS.get(name)
    if not spec:
        return False

    installer = _available_installer()
    installer_cmd = None
    if installer:
        installer_cmd = [installer]
    else:
        try:
            import importlib.util
            if importlib.util.find_spec("conda") is not None:
                installer_cmd = [sys.executable, "-m", "conda"]
        except Exception:
            installer_cmd = None

    if not installer_cmd:
        return False

    pkg, channel_flag, channel = spec
    cmd = installer_cmd + ["install", "-y", channel_flag, channel]
    prefix = _env_prefix()
    if prefix:
        cmd.extend(["-p", prefix])
    cmd.append(pkg)
    global _LAST_INSTALL_ERROR
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        _LAST_INSTALL_ERROR = None
    except Exception as exc:
        _LAST_INSTALL_ERROR = str(exc)
        if hasattr(exc, "stdout") and exc.stdout:
            _LAST_INSTALL_ERROR = exc.stdout
        return False
    return True

T = TypeVar("T")
R = TypeVar("R")

_LAST_INSTALL_ERROR: Optional[str] = None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def listify(val: str | Sequence[str]) -> List[str]:
    if isinstance(val, str):
        return [val]
    return list(val)


def resolve_executable(name: str, explicit: Optional[str] = None, auto_install: bool = False) -> str:
    """Resolve a CLI executable from PATH or current Python env bin."""
    if explicit:
        p = Path(explicit)
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
        raise FileNotFoundError(f"Executable not found or not executable: {explicit}")

    found = shutil.which(name)
    if found:
        return found

    env_bin = Path(sys.executable).parent
    for candidate in (env_bin / name, env_bin / f"{name}.exe"):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    if auto_install:
        if _install_tool(name):
            found = shutil.which(name)
            if found:
                return found
            env_bin = Path(sys.executable).parent
            for candidate in (env_bin / name, env_bin / f"{name}.exe"):
                if candidate.exists() and os.access(candidate, os.X_OK):
                    return str(candidate)

    hint = _INSTALL_HINTS.get(name)
    hint_msg = f" Try: `{hint}`" if hint else ""
    extra = f" Auto-install error: {_LAST_INSTALL_ERROR}" if _LAST_INSTALL_ERROR else ""
    raise FileNotFoundError(
        f"'{name}' not found on PATH or in the active environment bin.{hint_msg}{extra}"
    )


def build_env(
    extra_paths: Optional[Iterable[str]] = None,
    extra_env: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Build a subprocess environment with the current env bin on PATH."""
    env = os.environ.copy()
    env_bin = str(Path(sys.executable).parent)
    path_parts = [env_bin]
    if extra_paths:
        path_parts.extend([p for p in extra_paths if p])
    path_parts.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join(path_parts)
    if extra_env:
        env.update(extra_env)
    return env


def run_cmd(cmd: Sequence[str], env: Optional[dict[str, str]] = None,
            cwd: Optional[str] = None) -> None:
    """Run a command, streaming stdout/stderr, raise on failure."""
    print(">>", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
        env=env,
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end="")
    finally:
        proc.stdout.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}")


def is_gz(path: str | Path) -> bool:
    p = str(path)
    return p.endswith(".gz")


def parse_memory_bytes(value: str | int) -> int:
    """Parse memory strings like '8G', '8000M' into bytes."""
    if isinstance(value, int):
        return value
    raw = value.strip().upper()
    if raw.endswith("B"):
        raw = raw[:-1]
    units = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}
    if raw[-1] in units:
        num = float(raw[:-1])
        return int(num * units[raw[-1]])
    return int(float(raw))


def find_sra_file(srr: str, output_dir: str | Path) -> Optional[Path]:
    """Locate .sra/.sralite file for an SRR under output_dir."""
    output_dir = Path(output_dir)
    candidates = [
        output_dir / f"{srr}.sra",
        output_dir / f"{srr}.sralite",
        output_dir / srr / f"{srr}.sra",
        output_dir / srr / f"{srr}.sralite",
    ]
    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 0:
            return cand

    srr_dir = output_dir / srr
    if srr_dir.exists():
        hits = list(srr_dir.rglob(f"{srr}.sr*"))
        for hit in sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True):
            if hit.exists() and hit.stat().st_size > 0:
                return hit
    return None


def ensure_link(dest: Path, src: Path, prefer: str = "symlink") -> Path:
    """Ensure dest references src via symlink/hardlink or copy fallback."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    try:
        if prefer == "symlink":
            dest.symlink_to(src)
        elif prefer == "hardlink":
            os.link(src, dest)
        else:
            shutil.copy2(src, dest)
    except OSError:
        shutil.copy2(src, dest)
    return dest


def pick_compressor() -> tuple[str, List[str]]:
    """Return (executable, args) for pigz or gzip."""
    for name in ("pigz", "gzip"):
        try:
            exe = resolve_executable(name, auto_install=True)
            return exe, ["-f"]
        except FileNotFoundError:
            continue
    raise FileNotFoundError("Neither pigz nor gzip is available on PATH.")


def default_max_workers(n_items: int) -> int:
    cpu = os.cpu_count() or 4
    return max(1, min(n_items, max(1, cpu // 2)))


def resolve_jobs(n_items: int, jobs: Optional[int], max_workers: Optional[int]) -> int:
    """Resolve a job count with compatibility for legacy max_workers."""
    if jobs is None:
        jobs = max_workers
    if jobs is None:
        jobs = default_max_workers(n_items)
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    if n_items <= 0:
        return 1
    return min(jobs, n_items)


def run_in_threads(items: Sequence[T], worker: Callable[[T], R], max_workers: int) -> List[R]:
    if max_workers <= 1:
        return [worker(item) for item in items]

    results: List[Optional[R]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, item): idx for idx, item in enumerate(items)}
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
    return cast(List[R], results)
