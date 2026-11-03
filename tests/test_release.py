"""Release tooling used by the release workflow."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.portable

ROOT = Path(__file__).resolve().parents[1]


def notes(version):
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/release_notes.py"), version],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("tag", ["0.5.0", "v0.5.0"])
def test_release_notes_are_exactly_that_release_section(tag):
    result = notes(tag)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("- Extend portable training and serialization")
    assert result.stdout.rstrip().endswith("links for lyehe/torch-convnd.")
    assert "## " not in result.stdout


@pytest.mark.parametrize("tag", ["v9.9.9", "v0.5", "0.5.0.1"])
def test_release_notes_refuse_a_version_without_entries(tag):
    result = notes(tag)
    assert result.returncode != 0
    assert "has no entries" in result.stderr
