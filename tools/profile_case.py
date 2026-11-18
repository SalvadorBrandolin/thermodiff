"""Profile the generation of one case.

Usage: ``python tools/profile_case.py <model> <variant> [--workload W]
[--top 30] [--out file.prof]``.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import thermodiff  # noqa: E402
from thermodiff.bench_hooks import generate_workload  # noqa: E402

PHASE_KEYS = (
    "parse_time_s",
    "diff_time_s",
    "emit_time_s",
    "generation_time_s",
)


def generate(model: str, variant: str, workload: str) -> dict:
    """Generate one workload without writing it."""
    return generate_workload(
        model, workload, variant, tempfile.gettempdir(), write=False
    )


def print_phases(result: dict) -> None:
    """Print the per-phase times of a generation result."""
    for key in PHASE_KEYS:
        print(f"{key:20s} {result[key]:9.3f}")


def main(argv: list[str] | None = None) -> int:
    """Run the profiling CLI."""
    print(thermodiff.__file__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("variant")
    parser.add_argument("--workload", default="w6")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    print_phases(generate(args.model, args.variant, args.workload))
    profiler = cProfile.Profile()
    profiler.runcall(generate, args.model, args.variant, args.workload)
    if args.out:
        profiler.dump_stats(args.out)
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
