"""Regenerate a tier and compare it with the stored golden.

Usage: ``python tools/check_golden.py --tier fast|full|complete -j N``.
Exit status 1 if a file is missing, extra or different, or if a ``.model``
changed since the golden was written.
"""

from __future__ import annotations

import argparse
import difflib
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import thermodiff  # noqa: E402
from tools.golden import (  # noqa: E402
    _load_manifest,
    run_cases,
    sha256_file,
    write_golden,
)
from tools.golden_cases import (  # noqa: E402
    TIERS,
    Case,
    GoldenRoots,
    cases_for_tier,
)

SHOWN_DIFF_LINES = 30


def stale_models(
    cases: list[Case], roots: GoldenRoots, repo: Path
) -> set[str]:
    """Return the models whose ``.model`` hash differs from the manifest.

    Parameters
    ----------
    cases : list of Case
        Cases of the tier.
    roots : GoldenRoots
        Golden directories.
    repo : pathlib.Path
        Repository root.

    Returns
    -------
    set of str
        Names of the out-of-date models.
    """
    stale = set()
    for case in cases:
        model = case.model
        manifest = _load_manifest(roots.for_model(model))
        recorded = manifest["models"].get(model.name, {}).get("sha256")
        if recorded != sha256_file(repo / model.path):
            stale.add(model.name)
    return stale


def _unified_diff(case: Case, expected: str, actual: str) -> list[str]:
    """Return the unified diff lines (golden -> regenerated)."""
    return list(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            f"golden/{case.relpath}",
            f"generated/{case.relpath}",
            lineterm="",
        )
    )


def compare_cases(
    cases: list[Case],
    golden: GoldenRoots,
    generated: GoldenRoots,
) -> tuple[list[str], list[list[str]]]:
    """Compare regenerated files against the golden ones.

    Parameters
    ----------
    cases : list of Case
        Cases to compare (already filtered of stale models).
    golden : GoldenRoots
        Stored golden.
    generated : GoldenRoots
        Freshly regenerated files.

    Returns
    -------
    problems : list of str
        One line per missing or different file.
    diffs : list of list of str
        Unified diff of every differing file.
    """
    problems: list[str] = []
    diffs: list[list[str]] = []
    for case in cases:
        stored = golden.for_model(case.model) / case.relpath
        fresh = generated.for_model(case.model) / case.relpath
        if not stored.exists():
            problems.append(f"missing golden file: {case.relpath}")
            continue
        old = stored.read_text(encoding="utf-8")
        new = fresh.read_text(encoding="utf-8")
        if old != new:
            problems.append(f"different: {case.relpath}")
            diffs.append(_unified_diff(case, old, new))
    return problems, diffs


def extra_files(cases: list[Case], golden: GoldenRoots) -> list[str]:
    """Return golden files of the tier's models absent from the manifest.

    Parameters
    ----------
    cases : list of Case
        Cases of the tier.
    golden : GoldenRoots
        Stored golden.

    Returns
    -------
    list of str
        Messages, one per extra file.
    """
    extras = []
    for model in sorted({case.model for case in cases}):
        root = golden.for_model(model)
        listed = _load_manifest(root)["files"]
        for path in sorted((root / model.name).rglob("*.f90")):
            relpath = path.relative_to(root).as_posix()
            if relpath not in listed:
                extras.append(f"extra golden file: {relpath}")
    return extras


def _log_path(tier: str, repo: Path) -> Path:
    """Return where the full diffs are written."""
    logs = repo / ".orchestration" / "logs"
    if logs.parent.exists():
        logs.mkdir(parents=True, exist_ok=True)
        return logs / f"check_golden-{tier}.log"
    handle = tempfile.NamedTemporaryFile(
        prefix=f"check_golden-{tier}-", suffix=".log", delete=False
    )
    handle.close()
    return Path(handle.name)


def check_tier(
    tier: str,
    jobs: int,
    golden: GoldenRoots | None = None,
    repo: Path = REPO_ROOT,
) -> int:
    """Check a tier against the golden and report.

    Parameters
    ----------
    tier : str
        Tier name.
    jobs : int
        Number of worker processes.
    golden : GoldenRoots, optional
        Stored golden; defaults to :meth:`GoldenRoots.default`.
    repo : pathlib.Path, optional
        Repository root.

    Returns
    -------
    int
        Process exit status (0 = identical).
    """
    golden = golden or GoldenRoots.default(repo)
    cases = cases_for_tier(tier)
    problems = []
    stale = stale_models(cases, golden, repo)
    problems += [f"golden out of date for {name}" for name in sorted(stale)]
    cases = [case for case in cases if case.model.name not in stale]
    with tempfile.TemporaryDirectory() as tmp:
        generated = GoldenRoots(Path(tmp) / "examples", Path(tmp) / "ext")
        write_golden(cases, run_cases(cases, jobs), generated, repo)
        found, diffs = compare_cases(cases, golden, generated)
    problems += found + extra_files(cases, golden)
    return _report(tier, len(cases), problems, diffs, repo)


def _report(
    tier: str,
    n_cases: int,
    problems: list[str],
    diffs: list[list[str]],
    repo: Path,
) -> int:
    """Print the outcome and return the exit status."""
    if not problems:
        print(f"check_golden {tier}: OK ({n_cases} cases identical)")
        return 0
    log = _log_path(tier, repo)
    lines = problems + [line for diff in diffs for line in diff]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(problems[:SHOWN_DIFF_LINES]))
    if diffs:
        print("\n".join(diffs[0][:SHOWN_DIFF_LINES]))
    print(f"check_golden {tier}: {len(problems)} problem(s); log: {log}")
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the check CLI."""
    print(thermodiff.__file__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("-j", "--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    return check_tier(args.tier, args.jobs)


if __name__ == "__main__":
    sys.exit(main())
