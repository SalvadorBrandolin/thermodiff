"""Shared gfortran helpers for the test-suite."""

import os
import shutil
import subprocess
from pathlib import Path

EXECUTABLE_NAME = "a.exe" if os.name == "nt" else "a.out"
_STDERR_TAIL_CHARS = 1500


def find_gfortran() -> str | None:
    """Return the gfortran executable to use, honoring ``FC``.

    Returns
    -------
    str or None
        ``FC`` when its basename contains ``gfortran``, otherwise the
        ``gfortran`` found on ``PATH`` (``None`` when there is none).
    """
    compiler = os.environ.get("FC")
    if compiler and "gfortran" in os.path.basename(compiler).lower():
        return compiler
    return shutil.which("gfortran")


GFORTRAN = find_gfortran()


def compile_and_run(
    sources: list[Path],
    workdir: Path,
    flags: tuple[str, ...],
) -> subprocess.CompletedProcess:
    """Compile Fortran sources with gfortran and run the executable.

    Parameters
    ----------
    sources : list of Path
        Source files, compiled together in the given order.
    workdir : Path
        Working directory of both the compiler and the executable.
    flags : tuple of str
        Extra compiler flags (for example ``("-O2",)``).

    Returns
    -------
    subprocess.CompletedProcess
        The finished run of the executable (text mode, captured output).

    Raises
    ------
    AssertionError
        If compilation or execution exits with a non-zero status; the
        message carries the compiler or program stderr.
    """
    executable = workdir / EXECUTABLE_NAME
    compile_result = subprocess.run(
        [
            GFORTRAN,
            *flags,
            *(str(source) for source in sources),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        cwd=workdir,
    )
    assert compile_result.returncode == 0, (
        "COMPILE FAIL:\n" + compile_result.stderr[-_STDERR_TAIL_CHARS:]
    )
    run_result = subprocess.run(
        [str(executable)], capture_output=True, text=True, cwd=workdir
    )
    assert run_result.returncode == 0, (
        "RUN FAIL:\n" + run_result.stderr[-_STDERR_TAIL_CHARS:]
    )
    return run_result
