"""Vendored Bonsai-scout tree layout (upstream v1.0.0), CC BY-NC 4.0.

Only ``my_tree_layout`` is carried, not the Dash application around it. The file
is self-contained upstream (stdlib + numpy/pandas, no Bonsai imports);
:func:`omicverse.external.bonsai.scout_layout` builds its ``Layout_Tree``
straight from an edge list, so no ``SCData`` object is needed.

Two upstream defects are fixed here. Both need a tree whose subtrees differ in
depth, which is why upstream's own balanced-binary example never trips them and
a 400-cell PBMC tree trips both:

* ``equalDaylightAll`` reverts to ``old_tree`` when a step overshoots into
  negative daylight, but only assigns it after a *successful* step -- so a run
  that overshoots on step zero raised ``UnboundLocalError``. It is now seeded
  with the incoming equal-angle tree, which is what reverting means there.
* ``get_ed_angles`` computes ``child.opt_angle`` only on the branch that has
  shade to balance; the other branch left it ``None`` and the loop below it did
  ``opt_angle - angle`` -> ``TypeError``. With no shade to balance there is
  nothing to correct, so the optimal angle is set to the current one.

Both edits carry an ``omicverse:`` comment at the site. Nothing else is touched.
"""
