"""Command line interface for the model front-end.

Usable as ``thermodiff-model`` or ``python -m thermodiff.model``.
"""

import argparse
import sys

from thermodiff.model.emit import emit_module
from thermodiff.model.errors import ModelError
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

__all__ = ["main"]


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="thermodiff-model",
        description=(
            "Compile a .model file into a Fortran module with the model "
            "function and its analytical derivatives."
        ),
    )
    parser.add_argument(
        "model",
        help="path to the .model file, or '-' to read from stdin",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="path of the generated .f90 file (default: standard output)",
    )
    parser.add_argument(
        "--derivatives",
        help=(
            "derivatives to emit, for example 'dT,dn,dndn'.  Friendly "
            "names (dT, dV, dP, dn, dT2, dV2, dP2, dndn, dTdV, dTdP, "
            "dVdP, dndT, dndV, dndP) and the groups 'all', 'none', "
            "'first' and 'second' are accepted.  By default every "
            "derivative applicable to the declared variables is emitted."
        ),
    )
    parser.add_argument(
        "--optional-derivatives",
        action="store_true",
        help=(
            "declare every output as a Fortran 'optional' argument and "
            "compute it only when the caller passes it.  The generated "
            "subroutine can then be called with any subset of the "
            "derivatives."
        ),
    )
    return parser


def main(argv=None):
    """Run the command line interface.

    Parameters
    ----------
    argv : list of str, optional
        Argument list.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    args = _build_parser().parse_args(argv)

    derivatives = args.derivatives if args.derivatives else None

    try:
        if args.model == "-":
            text = sys.stdin.read()
            filename = "<stdin>"
        else:
            filename = args.model
            with open(filename, "r", encoding="utf-8") as handle:
                text = handle.read()
        raw = read_sections(text, filename)
        ir = build_ir(raw, filename)
        code = emit_module(
            ir,
            derivatives=derivatives,
            optional_derivatives=args.optional_derivatives,
        )
    except ModelError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for warning in ir.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(code)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(code)
    return 0
