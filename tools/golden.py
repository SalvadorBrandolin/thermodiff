"""Generate the golden Fortran files and their manifests.

Usage: ``python tools/golden.py [--tier fast|full|complete] [-j N]
[--out DIR]``.  With ``--out`` both golden roots are placed under ``DIR``
(``DIR/examples`` and ``DIR/external``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import thermodiff  # noqa: E402
from tools.golden_cases import (  # noqa: E402
    TIERS,
    Case,
    GoldenRoots,
    cases_for_tier,
)

MANIFEST_NAME = "manifest.json"
UNSET_SEED = "unset"
TIME_KEYS = (
    "parse_time_s",
    "diff_time_s",
    "emit_time_s",
    "generation_time_s",
)


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    return sha256_bytes(path.read_bytes())


def generate_case(case: Case, repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Generate one case without writing anything.

    Parameters
    ----------
    case : Case
        The ``(model, variant, workload)`` triple.
    repo : pathlib.Path, optional
        Repository root the model path is relative to.

    Returns
    -------
    dict
        ``source`` (file text) and the per-phase times.
    """
    from thermodiff.bench_hooks import generate_workload

    result = generate_workload(
        str(repo / case.model.path),
        case.workload,
        case.variant,
        tempfile.gettempdir(),
        write=False,
    )
    return {
        "source": result["source"],
        **{key: result[key] for key in TIME_KEYS},
    }


def _generator_info(repo: Path) -> dict[str, Any]:
    """Return the provenance block of the manifest."""
    import sympy

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "sympy": sympy.__version__,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", UNSET_SEED),
    }


def _load_manifest(root: Path) -> dict[str, Any]:
    """Return the manifest of a root, or an empty one."""
    path = root / MANIFEST_NAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"files": {}, "models": {}}


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    """Write a manifest with sorted keys."""
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (root / MANIFEST_NAME).write_text(text, encoding="utf-8")


def write_golden(
    cases: list[Case],
    results: list[dict[str, Any]],
    roots: GoldenRoots,
    repo: Path = REPO_ROOT,
) -> None:
    """Write the ``.f90`` files and merge them into the manifests.

    Parameters
    ----------
    cases : list of Case
        Generated cases.
    results : list of dict
        Output of :func:`generate_case`, aligned with ``cases``.
    roots : GoldenRoots
        Destination directories.
    repo : pathlib.Path, optional
        Repository root (model hashes and git provenance).
    """
    manifests: dict[Path, dict[str, Any]] = {}
    for case, result in zip(cases, results):
        root = roots.for_model(case.model)
        manifest = manifests.setdefault(root, _load_manifest(root))
        target = root / case.relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        data = result["source"].encode("utf-8")
        target.write_bytes(data)
        manifest["models"][case.model.name] = {
            "path": case.model.path,
            "sha256": sha256_file(repo / case.model.path),
        }
        manifest["files"][case.relpath] = {
            "sha256": sha256_bytes(data),
            **{key: round(result[key], 6) for key in TIME_KEYS},
        }
    info = _generator_info(repo)
    for root, manifest in manifests.items():
        manifest["generator"] = info
        _write_manifest(root, manifest)


def run_cases(cases: list[Case], jobs: int) -> list[dict[str, Any]]:
    """Generate cases in parallel, preserving order.

    Parameters
    ----------
    cases : list of Case
        Cases to generate.
    jobs : int
        Number of worker processes.

    Returns
    -------
    list of dict
        One :func:`generate_case` result per case.
    """
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(generate_case, cases, chunksize=1))


def roots_from_out(out: str | None) -> GoldenRoots:
    """Return the golden roots, relocated under ``out`` when given."""
    if out is None:
        return GoldenRoots.default()
    base = Path(out)
    return GoldenRoots(base / "examples", base / "external")


def main(argv: list[str] | None = None) -> int:
    """Run the golden generation CLI."""
    print(thermodiff.__file__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=TIERS, default="fast")
    parser.add_argument("-j", "--jobs", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    cases = cases_for_tier(args.tier)
    started = time.perf_counter()
    results = run_cases(cases, args.jobs)
    write_golden(cases, results, roots_from_out(args.out))
    elapsed = time.perf_counter() - started
    print(f"golden {args.tier}: {len(cases)} cases in {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
