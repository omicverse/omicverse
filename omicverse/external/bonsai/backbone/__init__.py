"""Vendored backbone-based Bonsai (upstream v1.0.0), CC BY-NC 4.0.

For datasets too large for a direct reconstruction: build a tree on a subset,
graft the remaining cells onto it, refine, repeat. Upstream's own four steps.

Modifications, all mechanical and marked ``omicverse:`` at the site:

* imports rewritten to relative form, and the ``sys.path``/``os.chdir``
  injections removed with them;
* each script's module-level body wrapped in ``main(args)``, with its
  ``ArgumentParser`` lifted into ``_build_parser()`` so both ``__main__`` and
  the in-process dispatcher can build the same namespace;
* ``backbone_based_bonsai`` dispatched its four steps by spawning
  ``python <script> <args>``. The scripts are vendored here, so ``_run_step``
  imports and calls them instead -- same steps, same arguments, one process.

The algorithm, its defaults and its numerics are upstream's.
"""
