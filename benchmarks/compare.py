"""Compile and run the generated Fortran, reporting a performance table.

The script materialises one source tree per *generation* of the code
generator (using ``git archive``, so the working copy is never touched),
asks that revision to emit the Fortran for a benchmark model, compiles it
with ``gfortran`` and times a subroutine call.

Usage
-----

::

    python benchmarks/compare.py
    python benchmarks/compare.py --nc 80 --reps 100 --models pr

The result is written to ``benchmarks/report.md``.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = Path(__file__).resolve().parent / "generate_model.py"

#: label -> git revision (``None`` means "the current working tree").
#:
#: ``mvp`` is the very first code generator: it cannot compile the
#: two-index Peng-Robinson model, which is itself part of the story.
VARIANTS = [
    ("mvp", "cf4f841"),
    ("no-share", "b30ddb8"),
    ("level-1", "6bc9861"),
    ("level-2", None),
]

#: model name -> Fortran program that times one call of the subroutine.
DRIVERS = {
    "pr": """\
program bench
   use bench_mod
   implicit none
   integer, parameter :: nc = {nc}
   integer, parameter :: reps = {reps}
   real(8) :: V, T, R
   real(8) :: ac(nc), Tc(nc), b(nc), kij(nc, nc), n(nc)
   real(8) :: fv, dfdT, dfdV, dfdT2, dfdV2, dfdTdV
   real(8) :: dfdn(nc), dfdndT(nc), dfdndV(nc), dfdndn(nc, nc)
   real(8) :: checksum, elapsed
   integer :: i, j, rep
   integer(8) :: t0, t1, rate
   R = 8.31446261815324d0
   V = 2.5d0
   T = 320.0d0
   do i = 1, nc
      n(i) = 0.5d0 + 0.05d0*i
      ac(i) = 0.4d0 + 0.01d0*i
      Tc(i) = 300.0d0 + 10.0d0*i
      b(i) = 0.02d0 + 0.001d0*i
   end do
   do i = 1, nc
      do j = 1, nc
         kij(i, j) = 0.02d0*abs(i - j)
      end do
   end do
   call pr_derivs(nc, V, T, R, Tc, ac, b, kij, n, fv, dfdT, dfdV, dfdn, &
                  dfdT2, dfdV2, dfdndn, dfdTdV, dfdndT, dfdndV)
   checksum = fv + dfdT + dfdV + dfdT2 + dfdV2 + dfdTdV &
              + sum(dfdn) + sum(dfdndT) + sum(dfdndV) + sum(dfdndn)
   call system_clock(t0, rate)
   do rep = 1, reps
      call pr_derivs(nc, V, T, R, Tc, ac, b, kij, n, fv, dfdT, dfdV, dfdn, &
                     dfdT2, dfdV2, dfdndn, dfdTdV, dfdndT, dfdndV)
   end do
   call system_clock(t1)
   elapsed = real(t1 - t0, 8)/real(rate, 8)
   write(*,'(A,ES24.16)') 'checksum ', checksum
   write(*,'(A,F16.1)')  'per_call ', elapsed/real(reps, 8)*1.0d9
end program bench
""",
    "toy": """\
program bench
   use bench_mod
   implicit none
   integer, parameter :: nc = {nc}
   integer, parameter :: reps = {reps}
   real(8) :: V, T
   real(8) :: n(nc)
   real(8) :: fv, dfdT, dfdV, dfdndV
   real(8) :: dfdn(nc), dfdndT(nc), dfdndn(nc, nc)
   real(8) :: checksum, elapsed
   integer :: i, rep
   integer(8) :: t0, t1, rate
   V = 2.5d0
   T = 320.0d0
   do i = 1, nc
      n(i) = 0.5d0 + 0.05d0*i
   end do
   call f_derivs(nc, V, T, n, fv, dfdT, dfdV, dfdn, dfdndn, dfdndT, dfdndV)
   checksum = fv + dfdT + dfdV + dfdndV &
              + sum(dfdn) + sum(dfdndT) + sum(dfdndn)
   call system_clock(t0, rate)
   do rep = 1, reps
      call f_derivs(nc, V, T, n, fv, dfdT, dfdV, dfdn, dfdndn, dfdndT, dfdndV)
   end do
   call system_clock(t1)
   elapsed = real(t1 - t0, 8)/real(rate, 8)
   write(*,'(A,ES24.16)') 'checksum ', checksum
   write(*,'(A,F16.1)')  'per_call ', elapsed/real(reps, 8)*1.0d9
end program bench
""",
}


def find_gfortran():
    """Return the gfortran executable, honoring ``FC``."""
    compiler = os.environ.get("FC")
    if compiler and "gfortran" in os.path.basename(compiler).lower():
        return compiler
    return shutil.which("gfortran")


def git_output(*args):
    """Run a git command in the repository and return its stdout."""
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def materialize(revision, cache):
    """Return a directory holding the source tree at ``revision``."""
    if revision is None:
        return ROOT
    destination = cache / revision
    if destination.exists():
        return destination
    destination.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(ROOT), "archive", revision],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["tar", "-x", "-C", str(destination)],
        input=archive.stdout,
        check=True,
    )
    return destination


def generate(source, model):
    """Return the Fortran emitted by ``source`` for ``model``.

    Raises
    ------
    RuntimeError
        If the revision cannot build the model at all.
    """
    environment = dict(os.environ, PYTHONPATH=str(source))
    result = subprocess.run(
        [sys.executable, str(GENERATOR), model],
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        tail = result.stderr.strip().splitlines()
        message = tail[-1] if tail else "unknown error"
        raise RuntimeError(message)
    return result.stdout


def measure(code, model, nc, reps, runs, gfortran, workdir):
    """Compile ``code`` and return its metrics, or ``None`` on failure."""
    program = (
        "module bench_mod\ncontains\n"
        + code
        + "\nend module bench_mod\n"
        + DRIVERS[model].format(nc=nc, reps=reps)
    )
    source = workdir / "main.f90"
    source.write_text(program)
    executable = workdir / "bench"
    compiled = subprocess.run(
        [gfortran, "-O2", "-o", str(executable), str(source)],
        capture_output=True,
        text=True,
    )
    if compiled.returncode != 0:
        return None

    timings = []
    checksum = None
    for _ in range(runs):
        run = subprocess.run(
            [str(executable)], capture_output=True, text=True, check=True
        )
        for line in run.stdout.splitlines():
            if line.startswith("per_call"):
                timings.append(float(line.split()[1]))
            elif line.startswith("checksum"):
                checksum = float(line.split()[1])
    return {
        "lines": len(code.splitlines()),
        "bytes": len(code.encode("utf-8")),
        "loops": sum(
            1 for line in code.splitlines() if line.strip().startswith("do ")
        ),
        "sqrts": code.count("sqrt("),
        "ns": min(timings),
        "checksum": checksum,
    }


def benchmark(model, nc, reps, runs, gfortran, cache, workdir):
    """Benchmark every variant for one model."""
    results = []
    for label, revision in VARIANTS:
        row = {"label": label, "revision": revision, "status": "ok"}
        try:
            source = materialize(revision, cache)
            row["hash"] = (
                "working tree"
                if revision is None
                else git_output("rev-parse", "--short", revision)
            )
            code = generate(source, model)
        except (RuntimeError, subprocess.CalledProcessError) as error:
            row["status"] = f"generation failed: {error}"
            results.append(row)
            continue
        row["hash"] = row.get("hash", "working tree")
        case = workdir / f"{model}-{label}"
        shutil.rmtree(case, ignore_errors=True)
        case.mkdir(parents=True)
        metrics = measure(code, model, nc, reps, runs, gfortran, case)
        if metrics is None:
            row["status"] = "does not compile"
        else:
            row.update(metrics)
        results.append(row)
    return results


def format_report(all_results, nc, reps, runs, gfortran):
    """Render the Markdown report."""
    lines = [
        "# Code generation benchmark",
        "",
        "Generated by `python benchmarks/compare.py`.",
        "",
        f"* compiler: `{gfortran}` with `-O2`",
        f"* `nc = {nc}`, `reps = {reps}`, best of `{runs}` runs",
        "* times are nanoseconds per call of the full subroutine",
        "",
    ]
    for model, results in all_results.items():
        ok = [row for row in results if row.get("ns")]
        fastest = min((row["ns"] for row in ok), default=None)
        lines += [
            f"## `{model}` model",
            "",
            "| generation | revision | lines | bytes | do loops | `sqrt` | "
            "ns/call | speed-up | status |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in results:
            if row.get("ns"):
                speedup = f"{row['ns'] / fastest:.2f}x" if fastest else "-"
                cells = (
                    f"{row['lines']}",
                    f"{row['bytes']}",
                    f"{row['loops']}",
                    f"{row['sqrts']}",
                    f"{row['ns']:,.0f}",
                    speedup,
                )
            else:
                cells = ("-", "-", "-", "-", "-", "-")
            lines.append(
                "| {label} | {hash} | {lines} | {bytes} | {loops} | {sqrts} "
                "| {ns} | {speedup} | {status} |".format(
                    label=row["label"],
                    hash=row.get("hash", row["revision"] or "working tree"),
                    lines=cells[0],
                    bytes=cells[1],
                    loops=cells[2],
                    sqrts=cells[3],
                    ns=cells[4],
                    speedup=cells[5],
                    status=row["status"],
                )
            )
        lines.append("")
    lines += [
        "## Notes",
        "",
        "* The `toy` model has no shared index-dependent factor, so all four",
        "  generations produce essentially the same code and time; it is a",
        "  control case.",
        "* `mvp` cannot compile the `pr` model: its free-index analysis",
        "  declares scalar outputs as arrays and gets the rank of `k_ij`",
        "  wrong.",
        "* Sharing the summations (level 1) is the dominant win;",
        "  materialising the shared index-dependent factors (level 2) then",
        "  removes the repeated `sqrt` of the attractive term.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    """Run the benchmark and write ``benchmarks/report.md``."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--models",
        default="toy,pr",
        help="comma separated models to benchmark (default: toy,pr)",
    )
    parser.add_argument(
        "--nc", type=int, default=50, help="number of components"
    )
    parser.add_argument(
        "--reps", type=int, default=200, help="calls per timing run"
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="timing runs, the best one is kept"
    )
    args = parser.parse_args(argv)

    gfortran = find_gfortran()
    if gfortran is None:
        parser.error("gfortran not found; set FC to a gfortran executable")

    cache = Path(tempfile.mkdtemp(prefix="thermodiff-bench-"))
    workdir = cache / "work"
    workdir.mkdir()
    try:
        all_results = {}
        for model in args.models.split(","):
            model = model.strip()
            if model not in DRIVERS:
                parser.error(f"unknown model {model!r}")
            all_results[model] = benchmark(
                model, args.nc, args.reps, args.runs, gfortran, cache, workdir
            )
        report = format_report(
            all_results, args.nc, args.reps, args.runs, gfortran
        )
        print(report)
        (ROOT / "benchmarks" / "report.md").write_text(report)
    finally:
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
