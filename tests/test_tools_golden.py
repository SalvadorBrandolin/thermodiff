"""Tests of the golden generation and check tooling."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import check_golden, golden  # noqa: E402
from tools.golden_cases import (  # noqa: E402
    MODELS,
    Case,
    GoldenRoots,
    cases_for_tier,
)

MOLEFRAC = next(spec for spec in MODELS if spec.name == "molefrac")


@pytest.fixture
def golden_roots(tmp_path):
    """Generate one tiny case into ``tmp_path`` and return its roots."""
    roots = GoldenRoots(tmp_path / "examples", tmp_path / "ext")
    cases = [Case(MOLEFRAC, "e0", "g1")]
    golden.write_golden(cases, [golden.generate_case(cases[0])], roots)
    return roots, cases


def test_tiers_are_nested():
    fast = set(cases_for_tier("fast"))
    full = set(cases_for_tier("full"))
    assert fast <= full <= set(cases_for_tier("complete"))


def test_generation_is_deterministic(golden_roots):
    roots, cases = golden_roots
    again = golden.generate_case(cases[0])["source"]
    stored = (roots.examples / cases[0].relpath).read_text(encoding="utf-8")
    assert stored == again
    manifest = golden._load_manifest(roots.examples)
    assert cases[0].relpath in manifest["files"]


def test_check_accepts_identical_files(golden_roots, tmp_path):
    roots, cases = golden_roots
    other = GoldenRoots(tmp_path / "o_examples", tmp_path / "o_ext")
    golden.write_golden(cases, [golden.generate_case(cases[0])], other)
    problems, diffs = check_golden.compare_cases(cases, roots, other)
    assert not problems and not diffs
    assert not check_golden.extra_files(cases, roots)


def test_check_detects_modified_file(golden_roots, tmp_path):
    roots, cases = golden_roots
    other = GoldenRoots(tmp_path / "o_examples", tmp_path / "o_ext")
    golden.write_golden(cases, [golden.generate_case(cases[0])], other)
    stored = roots.examples / cases[0].relpath
    stored.write_text(stored.read_text() + "! tampered\n", encoding="utf-8")
    problems, diffs = check_golden.compare_cases(cases, roots, other)
    assert problems == [f"different: {cases[0].relpath}"]
    assert any("tampered" in line for line in diffs[0])


def test_check_detects_extra_and_stale(golden_roots):
    roots, cases = golden_roots
    (roots.examples / "molefrac" / "e0" / "stray.f90").write_text("x\n")
    assert check_golden.extra_files(cases, roots)
    assert not check_golden.stale_models(cases, roots, REPO_ROOT)
    manifest = golden._load_manifest(roots.examples)
    manifest["models"]["molefrac"]["sha256"] = "0"
    golden._write_manifest(roots.examples, manifest)
    assert check_golden.stale_models(cases, roots, REPO_ROOT) == {"molefrac"}
