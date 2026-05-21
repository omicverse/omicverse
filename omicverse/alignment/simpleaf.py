#!/usr/bin/env python3
"""
CLI Interface Wrapper - Run simpleaf via command-line with a function-style API

This module provides a Python interface that wraps the ``simpleaf`` command line
tool (from the COMBINE-lab ecosystem), so that the simpleaf / salmon /
alevin-fry single-cell RNA-seq quantification workflow can be driven from
Python without importing internal libraries.

``simpleaf`` itself orchestrates ``salmon`` (mapping) and ``alevin-fry``
(quantification). The resulting ``alevin-fry`` quantification directory is
loaded into an :class:`anndata.AnnData` object using ``pyroe.load_fry``.

This mirrors the shape of :mod:`omicverse.alignment.kb_api` (the kb-python /
kallisto wrapper) so the two scRNA-seq preprocessing backends expose a parallel
API:

* ``ov.alignment.single.ref`` / ``ov.alignment.single.count``  -> kb-python
* ``ov.alignment.simpleaf.index`` / ``ov.alignment.simpleaf.count`` -> simpleaf

Author: Claude Code
Created: 2026
"""

import os
import sys
import shlex
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Union
from .._registry import register_function
from ._cli_utils import build_env, ensure_dir, resolve_executable, run_cmd


# ---------------------------------------------------------------------------
# Executable / environment resolution
# ---------------------------------------------------------------------------

def _resolve_simpleaf(explicit: Optional[str] = None) -> str:
    """Resolve the ``simpleaf`` executable.

    Parameters
    ----------
    explicit:str|None, optional
        Explicit path to the ``simpleaf`` binary, or a directory containing it
        (for example the ``bin`` directory of a dedicated conda environment).

    Returns
    -------
    str
        Absolute path to the ``simpleaf`` executable.
    """
    # Allow passing a directory that contains the binary.
    if explicit and os.path.isdir(explicit):
        cand = os.path.join(explicit, 'simpleaf')
        if os.path.exists(cand) and os.access(cand, os.X_OK):
            return cand
        raise FileNotFoundError(f"'simpleaf' not found inside directory: {explicit}")

    try:
        return resolve_executable("simpleaf", explicit=explicit, auto_install=False)
    except FileNotFoundError:
        pass

    raise FileNotFoundError(
        "Could not find the 'simpleaf' executable on PATH or in the active "
        "environment bin. simpleaf, salmon and alevin-fry are bioconda "
        "packages; install them into a dedicated environment, e.g.:\n"
        "  mamba create -n simpleaf -c conda-forge -c bioconda "
        "simpleaf salmon alevin-fry\n"
        "then pass the environment's bin directory via the `simpleaf_bin` "
        "argument (or add it to PATH)."
    )


def _simpleaf_bin_dir(simpleaf_exe: str) -> str:
    """Return the directory containing the resolved simpleaf executable."""
    return os.path.dirname(os.path.abspath(simpleaf_exe))


def _resolve_af_home(alevin_fry_home: Optional[str]) -> str:
    """Resolve the ``ALEVIN_FRY_HOME`` configuration directory.

    simpleaf requires the ``ALEVIN_FRY_HOME`` environment variable to point at
    a directory it can use to cache tool paths, custom chemistries and barcode
    permit lists. If the caller does not supply one we fall back to the
    existing environment variable, otherwise to ``~/.alevin_fry_home``.
    """
    if alevin_fry_home:
        af_home = os.path.abspath(os.path.expanduser(alevin_fry_home))
    elif os.environ.get('ALEVIN_FRY_HOME'):
        af_home = os.path.abspath(os.path.expanduser(os.environ['ALEVIN_FRY_HOME']))
    else:
        af_home = os.path.abspath(os.path.expanduser('~/.alevin_fry_home'))
    ensure_dir(af_home)
    return af_home


def _simpleaf_env(simpleaf_exe: str, af_home: str) -> Dict[str, str]:
    """Build a subprocess environment for simpleaf.

    Ensures the directory holding ``simpleaf`` (and therefore the sibling
    ``salmon`` / ``alevin-fry`` binaries from the same conda env) is on
    ``PATH`` and that ``ALEVIN_FRY_HOME`` is exported.
    """
    bin_dir = _simpleaf_bin_dir(simpleaf_exe)
    return build_env(
        extra_paths=[bin_dir],
        extra_env={'ALEVIN_FRY_HOME': af_home},
    )


def _ensure_set_paths(simpleaf_exe: str, af_home: str, env: Dict[str, str],
                      alevin_fry: Optional[str] = None,
                      piscem: Optional[str] = None,
                      force: bool = False,
                      dry_run: bool = False) -> None:
    """Run ``simpleaf set-paths`` once to register the helper binaries.

    simpleaf caches the paths to its helper tools inside a JSON config file in
    ``ALEVIN_FRY_HOME``. If that config already exists we skip this step unless
    ``force`` is requested.

    The accepted ``set-paths`` flags vary by simpleaf version: recent releases
    (>= 0.20) only expose ``--alevin-fry``, ``--piscem`` (and ``--macs``) and
    discover ``salmon`` automatically from ``PATH``. We therefore only pass the
    flags the installed binary actually advertises, and always make sure the
    simpleaf env ``bin`` directory is on ``PATH`` (via ``env``) so the
    remaining tools resolve.

    Parameters
    ----------
    simpleaf_exe:str
        Path to the resolved ``simpleaf`` binary.
    af_home:str
        Resolved ``ALEVIN_FRY_HOME`` directory.
    env:dict[str,str]
        Subprocess environment (must already contain ``ALEVIN_FRY_HOME`` and
        the simpleaf env ``bin`` directory on ``PATH``).
    alevin_fry, piscem:str|None, optional
        Explicit paths to the helper executables. When ``None`` the sibling
        binary next to ``simpleaf`` is used if present, otherwise simpleaf
        searches ``PATH``.
    force:bool, optional
        Re-run ``set-paths`` even when a cached config already exists.
    dry_run:bool, optional
        Print the command instead of executing it.
    """
    config_path = os.path.join(af_home, 'simpleaf_info.json')
    if os.path.exists(config_path) and not force and not dry_run:
        print(f"[simpleaf] set-paths already configured ({config_path})", flush=True)
        return

    # Discover which flags this simpleaf build supports.
    try:
        import subprocess as _sp
        help_txt = _sp.run(
            [simpleaf_exe, 'set-paths', '--help'],
            capture_output=True, text=True, env=env,
        ).stdout
    except Exception:
        help_txt = ''

    bin_dir = _simpleaf_bin_dir(simpleaf_exe)
    cmd: List[str] = [simpleaf_exe, 'set-paths']

    if '--alevin-fry' in help_txt or not help_txt:
        if alevin_fry is None:
            cand = os.path.join(bin_dir, 'alevin-fry')
            if os.path.exists(cand):
                alevin_fry = cand
        if alevin_fry:
            cmd.extend(['--alevin-fry', alevin_fry])
    if '--piscem' in help_txt:
        if piscem is None:
            cand = os.path.join(bin_dir, 'piscem')
            if os.path.exists(cand):
                piscem = cand
        if piscem:
            cmd.extend(['--piscem', piscem])
    # Older simpleaf releases also accept --salmon / --pyroe.
    if '--salmon' in help_txt:
        cand = os.path.join(bin_dir, 'salmon')
        if os.path.exists(cand):
            cmd.extend(['--salmon', cand])
    if '--pyroe' in help_txt:
        pyroe_exe = shutil.which('pyroe')
        if pyroe_exe:
            cmd.extend(['--pyroe', pyroe_exe])

    if dry_run:
        print("[simpleaf] (dry-run) " +
              " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
        return

    print(f"[simpleaf] Registering tool paths via 'simpleaf set-paths' "
          f"(ALEVIN_FRY_HOME={af_home})", flush=True)
    run_cmd(cmd, env=env)


def _append_flag(cmd: List[str], flag: str,
                 value: Optional[Union[str, int, float, bool]],
                 as_bool: bool = False):
    """Append a CLI flag to ``cmd``.

    Parameters
    ----------
    cmd:list[str]
        Command list being built.
    flag:str
        Flag name (for example ``--threads``).
    value
        Flag value. Skipped entirely when ``None``.
    as_bool:bool, optional
        When ``True`` treat ``value`` as a switch: append only the flag (no
        value) if ``value`` is truthy.
    """
    if as_bool:
        if value:
            cmd.append(flag)
        return
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _normalize_list_arg(val: Optional[Union[str, List[str]]],
                        sep: str = ',') -> Optional[str]:
    """Join a list argument into a separator-delimited string."""
    if val is None:
        return None
    if isinstance(val, list):
        return sep.join(str(v) for v in val)
    return val


# ---------------------------------------------------------------------------
# simpleaf index
# ---------------------------------------------------------------------------

@register_function(
    aliases=['simpleaf构建索引', 'alignment simpleaf index', 'simpleaf index',
             'salmon alevin-fry ref'],
    category="alignment",
    description="Build a spliced+intron (splici) reference and salmon index from a genome FASTA and GTF using simpleaf index.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['ov.alignment.simpleaf.index(fasta="genome.fa", gtf="genes.gtf", rlen=91, output="./af_index")'],
    related=['alignment.simpleaf_count', 'alignment.ref']
)
def simpleaf_index(
    output: str,
    fasta: Optional[str] = None,
    gtf: Optional[str] = None,
    rlen: int = 91,
    ref_type: str = 'spliced+intronic',
    threads: int = 16,
    kmer_length: int = 31,
    ref_seq: Optional[str] = None,
    keep_duplicates: bool = False,
    dedup: bool = False,
    overwrite: bool = False,
    use_piscem: bool = False,
    simpleaf_bin: Optional[str] = None,
    alevin_fry_home: Optional[str] = None,
    dry_run: bool = False,
    **kwargs
) -> Dict[str, str]:
    """
    Build a splici reference and index via ``simpleaf index``.

    ``simpleaf index`` extracts a spliced + intronic ("splici") transcriptome
    from a genome FASTA + GTF annotation and builds a salmon (or piscem) index
    from it, together with a transcript-to-gene (``t2g``) map suitable for
    downstream USA-mode quantification.

    Parameters
    ----------
    output:str
        Output directory for the generated reference and index. simpleaf
        creates ``index/`` and ``ref/`` subdirectories beneath it.
    fasta:str|None, optional
        Genome FASTA file used to extract the splici transcriptome. Required
        unless ``ref_seq`` is supplied for direct indexing.
    gtf:str|None, optional
        GTF annotation file aligned with ``fasta``. Required when ``fasta`` is
        given.
    rlen:int, optional
        Read length used to determine the intronic flank length when building
        the splici reference (``--rlen``). Typically the biological read 2
        length (for example 91 for 10x v3).
    ref_type:str, optional
        Expanded reference type, ``'spliced+intronic'`` (splici, default) or
        ``'spliced+unspliced'`` (spliceu).
    threads:int, optional
        Number of worker threads (``--threads``).
    kmer_length:int, optional
        K-mer length for index construction (``--kmer-length``).
    ref_seq:str|None, optional
        FASTA file for *direct* indexing, bypassing splici extraction. When
        given, ``fasta``/``gtf``/``rlen`` are not used.
    keep_duplicates:bool, optional
        Retain duplicate sequences during indexing (``--keep-duplicates``).
    dedup:bool, optional
        Remove identical sequences during reference expansion (``--dedup``).
    overwrite:bool, optional
        Overwrite an existing output directory (``--overwrite``).
    use_piscem:bool, optional
        Build a piscem index instead of the default salmon index
        (``--use-piscem``).
    simpleaf_bin:str|None, optional
        Explicit path to the ``simpleaf`` executable, or the ``bin`` directory
        of the conda environment that contains it.
    alevin_fry_home:str|None, optional
        Directory used as ``ALEVIN_FRY_HOME``. Defaults to ``$ALEVIN_FRY_HOME``
        or ``~/.alevin_fry_home``.
    dry_run:bool, optional
        Build and print the command without executing it. Returns the command
        string under the ``'command'`` key.
    **kwargs
        Additional flags forwarded verbatim (keys are converted from
        ``snake_case`` to ``--kebab-case``; boolean values become switches).

    Returns
    -------
    dict[str,str]
        Metadata dictionary with the produced ``index_dir``, ``t2g_path`` and
        ``ref_dir``, the resolved ``output`` directory and the command string.
    """
    print(f"[simpleaf index] Starting index build (ref_type={ref_type})", flush=True)

    if ref_seq is None and (fasta is None or gtf is None):
        raise ValueError(
            "simpleaf_index requires either `ref_seq` (direct indexing) or "
            "both `fasta` and `gtf` (splici/spliceu reference)."
        )

    output = os.path.abspath(os.path.expanduser(output))
    ensure_dir(output)

    simpleaf_exe = _resolve_simpleaf(simpleaf_bin)
    af_home = _resolve_af_home(alevin_fry_home)
    env = _simpleaf_env(simpleaf_exe, af_home)

    cmd: List[str] = [simpleaf_exe, 'index']
    _append_flag(cmd, '--output', output)
    _append_flag(cmd, '--threads', threads)
    _append_flag(cmd, '--kmer-length', kmer_length)

    if ref_seq is not None:
        print(f"[simpleaf index] Direct indexing of: {ref_seq}", flush=True)
        _append_flag(cmd, '--ref-seq', os.path.abspath(os.path.expanduser(ref_seq)))
    else:
        _append_flag(cmd, '--ref-type', ref_type)
        _append_flag(cmd, '--fasta', os.path.abspath(os.path.expanduser(fasta)))
        _append_flag(cmd, '--gtf', os.path.abspath(os.path.expanduser(gtf)))
        _append_flag(cmd, '--rlen', rlen)
        _append_flag(cmd, '--dedup', dedup, as_bool=True)

    _append_flag(cmd, '--keep-duplicates', keep_duplicates, as_bool=True)
    _append_flag(cmd, '--overwrite', overwrite, as_bool=True)
    _append_flag(cmd, '--use-piscem', use_piscem, as_bool=True)

    # Pass-through extras.
    for key, value in kwargs.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            _append_flag(cmd, flag, value, as_bool=True)
        else:
            _append_flag(cmd, flag, value)

    cmd_str = " ".join(shlex.quote(str(c)) for c in cmd)

    index_dir = os.path.join(output, 'index')
    ref_dir = os.path.join(output, 'ref')
    # The t2g map produced by `simpleaf index` lives in the index directory.
    t2g_path = os.path.join(index_dir, 't2g_3col.tsv')

    if dry_run:
        print(f"[simpleaf index] (dry-run) {cmd_str}", flush=True)
        return {
            'command': cmd_str,
            'output': output,
            'index_dir': index_dir,
            'ref_dir': ref_dir,
            't2g_path': t2g_path,
            'alevin_fry_home': af_home,
            'dry_run': True,
        }

    _ensure_set_paths(simpleaf_exe, af_home, env)

    try:
        run_cmd(cmd, env=env)
        print("[simpleaf index] index build completed!", flush=True)
    except Exception as e:
        print(f"[simpleaf index] index build failed: {e}", file=sys.stderr)
        raise

    # The 3-column t2g (with USA splice status) is preferred; fall back to the
    # plain 2-column map if the splici layout differs by version.
    if not os.path.exists(t2g_path):
        for cand in ('t2g_3col.tsv', 't2g.tsv'):
            cpath = os.path.join(index_dir, cand)
            if os.path.exists(cpath):
                t2g_path = cpath
                break

    result: Dict[str, str] = {
        'output': output,
        'index_dir': index_dir,
        'ref_dir': ref_dir,
        't2g_path': t2g_path,
        'ref_type': ref_type,
        'alevin_fry_home': af_home,
        'command': cmd_str,
    }
    return result


# ---------------------------------------------------------------------------
# simpleaf quant (count)
# ---------------------------------------------------------------------------

# friendly alias -> pyroe.load_fry output_format. The valid pyroe formats
# are 'scRNA', 'snRNA', 'S+A', 'U+S+A', 'velocity' and 'all'.
_OUTPUT_FORMAT_ALIASES = {
    'scrna': 'scRNA',
    'sc': 'scRNA',
    's+a': 'S+A',
    'snrna': 'snRNA',
    'sn': 'snRNA',
    'u+s+a': 'U+S+A',
    'usa': 'U+S+A',
    'velocity': 'velocity',
    'scvelo': 'velocity',
    'all': 'all',
}


@register_function(
    aliases=['simpleaf定量', 'alignment simpleaf count', 'simpleaf quant',
             'salmon alevin-fry count'],
    category="alignment",
    description="Map and quantify scRNA-seq FASTQs against a simpleaf index with salmon + alevin-fry, then load the result into an AnnData h5ad.",
    prerequisites={'functions': ['simpleaf_index']},
    requires={},
    produces={},
    auto_fix='escalate',
    examples=['ov.alignment.simpleaf.count(index="./af_index/index", t2g_map="./af_index/index/t2g_3col.tsv", chemistry="10xv3", reads1=["R1.fastq.gz"], reads2=["R2.fastq.gz"], output="./af_quant")'],
    related=['alignment.simpleaf_index', 'alignment.count']
)
def simpleaf_count(
    index: str,
    reads1: Union[str, List[str]],
    reads2: Union[str, List[str]],
    t2g_map: str,
    output: str = '.',
    chemistry: str = '10xv3',
    resolution: str = 'cr-like',
    expected_ori: str = 'both',
    threads: int = 16,
    unfiltered_pl: Union[bool, str] = True,
    explicit_pl: Optional[str] = None,
    expect_cells: Optional[int] = None,
    forced_cells: Optional[int] = None,
    knee: bool = False,
    min_reads: Optional[int] = None,
    use_piscem: bool = False,
    anndata_out: bool = False,
    h5ad_path: Optional[str] = None,
    output_format: str = 'scRNA',
    simpleaf_bin: Optional[str] = None,
    alevin_fry_home: Optional[str] = None,
    dry_run: bool = False,
    **kwargs
) -> Dict[str, str]:
    """
    Map and quantify FASTQ data via ``simpleaf quant`` and load into AnnData.

    ``simpleaf quant`` runs the full single-cell quantification pipeline:
    salmon (or piscem) mapping, ``alevin-fry generate-permit-list``,
    ``collate`` and ``quant``. The resulting ``alevin-fry`` quantification
    directory (``af_quant/alevin``) is then loaded into an
    :class:`anndata.AnnData` with ``pyroe.load_fry`` and written to an
    ``.h5ad`` file.

    Parameters
    ----------
    index:str
        Path to the simpleaf/salmon index directory produced by
        :func:`simpleaf_index` (the ``index`` subdirectory of its output).
    reads1:str|list[str]
        Read 1 (barcode+UMI) FASTQ file(s) (``--reads1``).
    reads2:str|list[str]
        Read 2 (cDNA) FASTQ file(s) (``--reads2``).
    t2g_map:str
        Transcript-to-gene mapping file produced by :func:`simpleaf_index`
        (``--t2g-map``), normally the 3-column USA-mode map.
    output:str, optional
        Output directory for the simpleaf quant results.
    chemistry:str, optional
        Single-cell chemistry, for example ``'10xv2'``, ``'10xv3'``,
        ``'10xv4-3p'`` or a custom geometry string (``--chemistry``).
    resolution:str, optional
        UMI resolution strategy (``--resolution``): one of ``'cr-like'``,
        ``'cr-like-em'``, ``'parsimony'``, ``'parsimony-em'``,
        ``'parsimony-gene'``, ``'parsimony-gene-em'``.
    expected_ori:str, optional
        Expected read orientation relative to the transcript
        (``--expected-ori``): ``'fw'``, ``'rc'`` or ``'both'``.
    threads:int, optional
        Number of worker threads (``--threads``).
    unfiltered_pl:bool|str, optional
        Use an unfiltered permit list (``--unfiltered-pl``). ``True`` lets
        simpleaf fetch the standard 10x barcode list automatically; a string
        is treated as an explicit path to an unfiltered permit-list file.
        Mutually exclusive with ``explicit_pl`` / ``knee`` / ``expect_cells`` /
        ``forced_cells``.
    explicit_pl:str|None, optional
        Path to an explicit *filtered* permit list (``--explicit-pl``).
    expect_cells:int|None, optional
        Expected number of cells (``--expect-cells``).
    forced_cells:int|None, optional
        Force a fixed number of cells (``--forced-cells``).
    knee:bool, optional
        Use knee-point permit-list filtering (``--knee``).
    min_reads:int|None, optional
        Minimum number of reads for a barcode to be considered (``--min-reads``).
    use_piscem:bool, optional
        Use piscem for mapping instead of salmon (``--use-piscem``).
    anndata_out:bool, optional
        Also ask ``simpleaf quant`` itself to emit an ``.h5ad`` via its
        ``--anndata-out`` flag, in addition to the pyroe-loaded output.
    h5ad_path:str|None, optional
        Destination path for the AnnData ``.h5ad`` written by this wrapper.
        Defaults to ``<output>/adata.h5ad``.
    output_format:str, optional
        Layout passed to ``pyroe.load_fry``. Accepts ``'scRNA'`` (spliced +
        ambiguous counts in ``X``, the default — standard single-cell),
        ``'snRNA'`` (single-nucleus), ``'velocity'`` (separate spliced /
        unspliced layers for RNA-velocity), ``'S+A'``, ``'U+S+A'`` and
        ``'all'`` (case-insensitive).
    simpleaf_bin:str|None, optional
        Explicit path to the ``simpleaf`` executable, or the ``bin`` directory
        of the conda environment that contains it.
    alevin_fry_home:str|None, optional
        Directory used as ``ALEVIN_FRY_HOME``.
    dry_run:bool, optional
        Build and print the command without executing it.
    **kwargs
        Additional flags forwarded verbatim (``snake_case`` -> ``--kebab-case``).

    Returns
    -------
    dict[str,str]
        Metadata dictionary including the ``quant_dir`` (alevin-fry output),
        the ``h5ad_file`` written via pyroe, the resolved ``output`` directory
        and the command string.
    """
    print(f"[simpleaf quant] Starting quantification (chemistry={chemistry}, "
          f"resolution={resolution})", flush=True)

    output = os.path.abspath(os.path.expanduser(output))
    ensure_dir(output)

    simpleaf_exe = _resolve_simpleaf(simpleaf_bin)
    af_home = _resolve_af_home(alevin_fry_home)
    env = _simpleaf_env(simpleaf_exe, af_home)

    reads1_joined = _normalize_list_arg(
        [os.path.abspath(os.path.expanduser(r)) for r in
         (reads1 if isinstance(reads1, list) else [reads1])])
    reads2_joined = _normalize_list_arg(
        [os.path.abspath(os.path.expanduser(r)) for r in
         (reads2 if isinstance(reads2, list) else [reads2])])

    cmd: List[str] = [simpleaf_exe, 'quant']
    _append_flag(cmd, '--index', os.path.abspath(os.path.expanduser(index)))
    _append_flag(cmd, '--reads1', reads1_joined)
    _append_flag(cmd, '--reads2', reads2_joined)
    _append_flag(cmd, '--t2g-map', os.path.abspath(os.path.expanduser(t2g_map)))
    _append_flag(cmd, '--output', output)
    _append_flag(cmd, '--chemistry', chemistry)
    _append_flag(cmd, '--resolution', resolution)
    _append_flag(cmd, '--expected-ori', expected_ori)
    _append_flag(cmd, '--threads', threads)
    _append_flag(cmd, '--use-piscem', use_piscem, as_bool=True)
    _append_flag(cmd, '--anndata-out', anndata_out, as_bool=True)

    # Permit-list strategy: exactly one of these should be active.
    pl_modes = sum([
        bool(unfiltered_pl),
        explicit_pl is not None,
        expect_cells is not None,
        forced_cells is not None,
        bool(knee),
    ])
    if pl_modes > 1:
        print("[simpleaf quant] Warning: multiple permit-list strategies "
              "requested; simpleaf may reject this combination.", flush=True)
    if explicit_pl is not None:
        _append_flag(cmd, '--explicit-pl',
                     os.path.abspath(os.path.expanduser(explicit_pl)))
    elif expect_cells is not None:
        _append_flag(cmd, '--expect-cells', expect_cells)
    elif forced_cells is not None:
        _append_flag(cmd, '--forced-cells', forced_cells)
    elif knee:
        _append_flag(cmd, '--knee', True, as_bool=True)
    elif unfiltered_pl:
        if isinstance(unfiltered_pl, str):
            # Explicit path to an unfiltered permit list.
            cmd.extend(['--unfiltered-pl',
                        os.path.abspath(os.path.expanduser(unfiltered_pl))])
        else:
            # Bare flag: simpleaf auto-fetches the standard 10x permit list.
            cmd.append('--unfiltered-pl')

    _append_flag(cmd, '--min-reads', min_reads)

    # Pass-through extras.
    for key, value in kwargs.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            _append_flag(cmd, flag, value, as_bool=True)
        else:
            _append_flag(cmd, flag, value)

    cmd_str = " ".join(shlex.quote(str(c)) for c in cmd)

    # alevin-fry writes its quantification under <output>/af_quant.
    quant_dir = os.path.join(output, 'af_quant')
    if h5ad_path is None:
        h5ad_path = os.path.join(output, 'adata.h5ad')
    else:
        h5ad_path = os.path.abspath(os.path.expanduser(h5ad_path))

    fmt = _OUTPUT_FORMAT_ALIASES.get(str(output_format).lower(), output_format)

    if dry_run:
        print(f"[simpleaf quant] (dry-run) {cmd_str}", flush=True)
        return {
            'command': cmd_str,
            'output': output,
            'quant_dir': quant_dir,
            'h5ad_file': h5ad_path,
            'output_format': fmt,
            'alevin_fry_home': af_home,
            'dry_run': True,
        }

    _ensure_set_paths(simpleaf_exe, af_home, env)

    try:
        run_cmd(cmd, env=env)
        print("[simpleaf quant] quantification completed!", flush=True)
    except Exception as e:
        print(f"[simpleaf quant] quantification failed: {e}", file=sys.stderr)
        raise

    result: Dict[str, str] = {
        'output': output,
        'quant_dir': quant_dir,
        'chemistry': chemistry,
        'resolution': resolution,
        'output_format': fmt,
        'alevin_fry_home': af_home,
        'command': cmd_str,
    }

    # Load the alevin-fry quant directory into an AnnData via pyroe.
    try:
        import pyroe
    except ImportError:
        print("[simpleaf quant] Warning: 'pyroe' is not installed; skipping "
              "AnnData conversion. Install with: pip install pyroe", flush=True)
        return result

    # pyroe.load_fry expects the directory that contains quants_mat*.* and the
    # quant.json sidecar - that is the af_quant directory itself.
    frydir = quant_dir
    if not os.path.isdir(frydir):
        print(f"[simpleaf quant] Warning: expected quant directory not found: "
              f"{frydir}; skipping AnnData conversion.", flush=True)
        return result

    print(f"[simpleaf quant] Loading af_quant into AnnData "
          f"(output_format={fmt}) ...", flush=True)
    try:
        adata = pyroe.load_fry(frydir, output_format=fmt)
        ensure_dir(os.path.dirname(h5ad_path) or '.')
        adata.write_h5ad(h5ad_path)
        result['h5ad_file'] = h5ad_path
        result['n_obs'] = str(adata.n_obs)
        result['n_vars'] = str(adata.n_vars)
        print(f"[simpleaf quant] Wrote AnnData: {h5ad_path} "
              f"({adata.n_obs} cells x {adata.n_vars} genes)", flush=True)
    except Exception as e:
        print(f"[simpleaf quant] Warning: pyroe.load_fry / h5ad write failed: "
              f"{e}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Convenience: one-click index + quant
# ---------------------------------------------------------------------------

def simpleaf_pipeline(
    fasta: str,
    gtf: str,
    reads1: Union[str, List[str]],
    reads2: Union[str, List[str]],
    index_output: str = 'af_index',
    quant_output: str = 'af_quant_out',
    chemistry: str = '10xv3',
    rlen: int = 91,
    threads: int = 16,
    output_format: str = 'scRNA',
    **kwargs
) -> Dict[str, Dict[str, str]]:
    """
    One-click simpleaf workflow: build a splici index then map+quantify FASTQs.

    Parameters
    ----------
    fasta:str
        Genome FASTA file.
    gtf:str
        GTF annotation file.
    reads1, reads2:str|list[str]
        Read 1 / read 2 FASTQ files.
    index_output:str, optional
        Output directory for :func:`simpleaf_index`.
    quant_output:str, optional
        Output directory for :func:`simpleaf_count`.
    chemistry:str, optional
        Single-cell chemistry passed to :func:`simpleaf_count`.
    rlen:int, optional
        Read length used for splici intron flanking.
    threads:int, optional
        Worker threads for both steps.
    output_format:str, optional
        ``pyroe.load_fry`` output layout.
    **kwargs
        Extra keyword arguments forwarded to :func:`simpleaf_count`.

    Returns
    -------
    dict[str,dict[str,str]]
        Mapping with ``'index'`` and ``'count'`` result dictionaries.
    """
    results: Dict[str, Dict[str, str]] = {}
    print("[simpleaf] Step 1/2: building splici index ...", flush=True)
    idx = simpleaf_index(
        output=index_output, fasta=fasta, gtf=gtf, rlen=rlen, threads=threads,
    )
    results['index'] = idx

    print("[simpleaf] Step 2/2: mapping + quantifying FASTQs ...", flush=True)
    cnt = simpleaf_count(
        index=idx['index_dir'],
        reads1=reads1,
        reads2=reads2,
        t2g_map=idx['t2g_path'],
        output=quant_output,
        chemistry=chemistry,
        threads=threads,
        output_format=output_format,
        **kwargs
    )
    results['count'] = cnt
    return results


# ---------------------------------------------------------------------------
# Namespace object (parallels ``ov.alignment.single`` from kb_api)
# ---------------------------------------------------------------------------

import types
simpleaf = types.SimpleNamespace()
simpleaf.index = simpleaf_index
simpleaf.count = simpleaf_count
simpleaf.pipeline = simpleaf_pipeline
