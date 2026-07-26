r"""Vendored PDBFixer — structure repair for molecular-dynamics preparation.

`PDBFixer <https://github.com/openmm/pdbfixer>`_ (Peter Eastman, Stanford /
Simbios) repairs experimental and predicted structures so they can actually be
simulated: it builds missing residues and heavy atoms from residue templates,
replaces non-standard residues, strips heterogens, and adds hydrogens at a
requested pH.

Why it is vendored
------------------
PDBFixer is **not published on PyPI** (upstream ships it through conda-forge
only), so ``pip install 'omicverse[md]'`` could never pull it in. Since it is
pure Python and depends only on OpenMM + NumPy — both of which the MD extra
already requires — it is bundled here so :func:`omicverse.mol.prepare_system`
works from a plain pip install. The upstream conda package is not needed.

Local modifications (kept deliberately minimal)
-----------------------------------------------
* removed the deprecated ``from pkg_resources import resource_filename``
  import (unused; ``pkg_resources`` is slated for removal from setuptools);
* removed the command-line ``main()`` entry point, which imported the
  optional web UI (``pdbfixer.ui``) that is not shipped here;
* in ``addMissingAtoms``, pinned the short minimisation of newly built atoms
  to OpenMM's ``Reference`` platform. Upstream lets OpenMM choose the fastest
  platform, which is normally a GPU, and GPU force reductions are not
  order-deterministic — so repaired coordinates differed between runs even
  with a fixed seed, making every downstream simulation irreproducible. Only
  the handful of added atoms is involved, so the slower platform costs little.

Everything else — the residue templates in ``templates/``, ``soft.xml``, and
the :class:`PDBFixer` algorithm itself — is upstream v1.8.1, unmodified.

Reproducibility note
--------------------
:func:`omicverse.mol.prepare_system` additionally seeds Python's and NumPy's
global RNGs around this code, because OpenMM's ``Modeller.addHydrogens`` and
``addSolvent`` draw from them rather than accepting a seed.

License
-------
MIT, Portions copyright (c) 2013 Stanford University and the Authors
(Peter Eastman et al.). Full text in ``LICENSE`` next to this file.

If you use this in published work, cite OpenMM/PDBFixer:
Eastman et al., *OpenMM 7*, PLoS Comput Biol 13(7): e1005659 (2017).
"""

from __future__ import annotations

__all__ = ["PDBFixer"]

# Upstream version this copy was taken from.
__vendored_version__ = "1.8.1"


def __getattr__(name: str):
    """Import :class:`PDBFixer` lazily.

    ``pdbfixer.py`` imports OpenMM at module scope, so resolving it eagerly
    would make ``import omicverse.external.pdbfixer`` fail whenever the ``[md]``
    extra is absent — and by extension break any module that merely *mentions*
    this package. Deferring the import keeps the package importable everywhere
    and turns a missing backend into the actionable error raised by
    :func:`omicverse.mol.md._common.check_md`.
    """
    if name == "PDBFixer":
        from .pdbfixer import PDBFixer
        return PDBFixer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals().keys()) + __all__))
