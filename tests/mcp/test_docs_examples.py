"""Tests to verify documentation stays consistent with code."""

import pathlib

# Resolve project root (tests/mcp/ -> project root)
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


class TestDocsConsistency:
    def test_phase_names_consistent(self):
        """Quickstart should reference all three phase names."""
        quickstart = (_PROJECT_ROOT / "docs" / "mcp_quickstart.md").read_text()
        assert "P0" in quickstart
        assert "P0.5" in quickstart or "P0+P0.5" in quickstart
        assert "P2" in quickstart

    def test_quickstart_mentions_core_meta_capabilities(self):
        """Quickstart should mention representative meta-tool capabilities."""
        quickstart = (_PROJECT_ROOT / "docs" / "mcp_quickstart.md").read_text()
        for name in ("ov.list_tools", "ov.describe_tool", "ov.get_session", "ov.get_health"):
            assert name in quickstart

    def test_readme_points_to_agent_docs(self):
        """The README links to OmicClaw, which documents the MCP server and agents."""
        readme = (_PROJECT_ROOT / "README.md").read_text()
        assert "github.com/omicverse/omicclaw" in readme

    def test_cli_command_in_quickstart(self):
        quickstart = (_PROJECT_ROOT / "docs" / "mcp_quickstart.md").read_text()
        assert "python -m omicverse.mcp" in quickstart

    def test_entrypoint_in_pyproject(self):
        pyproject = (_PROJECT_ROOT / "pyproject.toml").read_text()
        assert "omicverse-mcp" in pyproject

    def test_mcp_extra_in_pyproject(self):
        pyproject = (_PROJECT_ROOT / "pyproject.toml").read_text()
        assert "mcp" in pyproject

    def test_integration_doc_exists(self):
        assert (_PROJECT_ROOT / "docs" / "mcp_integration.md").exists()
