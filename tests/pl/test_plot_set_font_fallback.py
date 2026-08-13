"""plot_set must keep CJK-capable fallback fonts in the rcParams chain."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import omicverse as ov
from omicverse.pl import _plot_backend as backend


def _fake_font_resolver(mapping):
    def resolve(font_path):
        return mapping.get(font_path)
    return resolve


def test_plot_set_prefers_primary_then_user_fallback(monkeypatch):
    monkeypatch.setattr(
        backend, "_resolve_font_name",
        _fake_font_resolver({"Primary": "Primary", "SimHei": "SimHei"}))
    monkeypatch.setattr(
        backend, "_available_font_families", lambda families: list(families))

    with matplotlib.rc_context():
        ov.pl.plot_set(
            scanpy=False, font_path="Primary",
            fallback_font_path="SimHei", show_monitor=False)
        chain = matplotlib.rcParams["font.sans-serif"]

    assert chain[0] == "Primary"
    assert "SimHei" in chain


def test_plot_set_falls_back_when_primary_font_is_missing(monkeypatch):
    monkeypatch.setattr(
        backend, "_resolve_font_name",
        _fake_font_resolver({"SimHei": "SimHei"}))
    monkeypatch.setattr(
        backend, "_available_font_families",
        lambda families: [f for f in families if f == "SimHei"])

    with matplotlib.rc_context():
        ov.pl.plot_set(
            scanpy=False, font_path="missing.ttf",
            fallback_font_path="SimHei", show_monitor=False)
        chain = matplotlib.rcParams["font.sans-serif"]

    assert "SimHei" in chain
    assert "missing.ttf" not in chain


def test_plot_set_scanpy_defaults_keep_cjk_fallback(monkeypatch):
    monkeypatch.setattr(
        backend, "_available_font_families",
        lambda families: [f for f in families if f == "SimHei"])

    with matplotlib.rc_context():
        ov.pl.plot_set(scanpy=True, show_monitor=False)
        chain = matplotlib.rcParams["font.sans-serif"]

    assert "SimHei" in chain
