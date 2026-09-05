from __future__ import annotations

from typing import get_args

import pytest

from omicverse.space.histo._dispatch import PredictMethod, predict_expression


def test_histo_predict_method_list_contains_no_ghost_backends():
    assert set(get_args(PredictMethod)) == {"stpath", "stflow", "hest_fm"}


def test_removed_bleep_backend_fails_at_dispatch_boundary():
    with pytest.raises(ValueError, match="stpath, stflow, hest_fm"):
        predict_expression(None, method="bleep")
