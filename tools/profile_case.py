"""Profile the generation of one case.

Usage: ``python tools/profile_case.py <model> <variant> [--workload W]
[--top 30] [--out file.prof] [--warm]``.

By default the profiled generation runs in a fresh child process, so its
diff phase is cold (SymPy's global ``cacheit`` cache starts empty).  With
``--warm`` the profiled generation runs in the same process after the
unprofiled one, which reproduces the behaviour of the tool before T1.12.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import thermodiff  # noqa: E402
from benchmarks.td_bench import generate_workload  # noqa: E402

PHASE_KEYS = (
    "parse_time_s",
    "diff_time_s",
    "emit_time_s",
    "generation_time_s",
)
CHILD_FLAG = "--_profile-child"
PROFILE_NAME = "profile.prof"
SORT_KEY = "cumulative"
PYTHONPATH_VAR = "PYTHONPATH"


def generate(model: str, variant: str, workload: str) -> dict:
    """Generate one workload without writing it."""
    return generate_workload(
        model, workload, variant, tempfile.gettempdir(), write=False
    )


def print_phases(result: dict) -> None:
    """Print the per-phase times of a generation result."""
    for key in PHASE_KEYS:
        print(f"{key:20s} {result[key]:9.3f}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line of the profiling CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("variant")
    parser.add_argument("--workload", default="w6")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--warm",
        action="store_true",
        help="profile a second generation in the same process (warm)",
    )
    parser.add_argument(
        CHILD_FLAG,
        dest="child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def profile_child(args: argparse.Namespace) -> None:
    """Profile one cold generation and dump the stats to ``args.out``."""
    profiler = cProfile.Profile()
    profiler.runcall(generate, args.model, args.variant, args.workload)
    profiler.dump_stats(args.out)


def run_cold_child(args: argparse.Namespace, out_path: str) -> None:
    """Run ``profile_child`` in a fresh interpreter and wait for it.

    The child gets the repository root on ``PYTHONPATH`` so that it imports
    the same ``thermodiff`` as the parent.
    """
    env = dict(os.environ)
    previous = env.get(PYTHONPATH_VAR, "")
    paths = [str(REPO_ROOT)] + ([previous] if previous else [])
    env[PYTHONPATH_VAR] = os.pathsep.join(paths)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        args.model,
        args.variant,
        "--workload",
        args.workload,
        "--out",
        out_path,
        CHILD_FLAG,
    ]
    subprocess.run(command, env=env, check=True)


def cold_stats(args: argparse.Namespace) -> pstats.Stats:
    """Return the stats of a cold generation run in a fresh process."""
    if args.out:
        run_cold_child(args, args.out)
        return pstats.Stats(args.out)
    with tempfile.TemporaryDirectory() as scratch:
        out_path = str(Path(scratch) / PROFILE_NAME)
        run_cold_child(args, out_path)
        return pstats.Stats(out_path)


def warm_stats(args: argparse.Namespace) -> pstats.Stats:
    """Return the stats of a generation profiled after a warm-up run."""
    profiler = cProfile.Profile()
    profiler.runcall(generate, args.model, args.variant, args.workload)
    if args.out:
        profiler.dump_stats(args.out)
    return pstats.Stats(profiler)


def main(argv: list[str] | None = None) -> int:
    """Run the profiling CLI."""
    args = parse_args(argv)
    if args.child:
        profile_child(args)
        return 0
    print(thermodiff.__file__)
    print_phases(generate(args.model, args.variant, args.workload))
    if args.warm:
        stats = warm_stats(args)
    else:
        stats = cold_stats(args)
    stats.sort_stats(SORT_KEY).print_stats(args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
