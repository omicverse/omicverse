"""Vendored Bonsai core (upstream v1.0.0), CC BY-NC 4.0.

Modifications from upstream, all mechanical:

* ``from bonsai.X import ...`` -> ``from .X import ...`` and
  ``import bonsai.X as Y`` -> ``from . import X as Y``, so the modules import as
  a package instead of relying on the script's own ``sys.path.append`` of its
  parent directory. The two ``sys.path`` injections are removed with them.
* ``bonsai_main`` had its module-level body wrapped in ``main(args)`` and its
  ``argparse`` block moved under ``__main__``, so omicverse can call it in
  process rather than spawning a Python subprocess.

Nothing else is touched: the algorithm, its defaults and its numerics are
upstream's.
"""
