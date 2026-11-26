"""Command line interface for the model front-end.

Usable as ``thermodiff-model`` or ``python -m thermodiff.model``.
"""

import argparse
import sys

from thermodiff.codegen.options import Preset
from thermodiff.model.emit import emit_module
from thermodiff.model.errors import ModelError
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

__all__ = ["main"]


def _parse_sizes(text):
    """Parse a comma separated list of component counts."""
    try:
        return tuple(int(item) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected integers separated by commas, got {text!r}"
        ) from error


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
        "--workload",
        action="append",
        help=(
            "workload mode: emit a fixed-output kernel for this derivative "
            "set, for example --workload dn,dndn --workload dT (repeat the "
            "flag once per set).  The module gets a dispatcher with the "
            "signature of the optional-derivatives kernel that picks the "
            "kernel of the exact set of present outputs and falls back to "
            "the optional kernel otherwise; --derivatives selects the "
            "outputs of that signature."
        ),
    )
    parser.add_argument(
        "--fixed-nc",
        type=_parse_sizes,
        default=None,
        help=(
            "with --workload: also emit each kernel for these compile-time "
            "numbers of components, for example 2,3; the dispatcher "
            "selects by nc."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=[preset.value for preset in Preset],
        default=Preset.STANDARD.value,
        help=(
            "named set of code-generation options (default: %(default)s). "
            "minimal: the plain generator.  standard: the recommended "
            "set.  fastest: the fastest generated code, with slower "
            "generation and last-bit rounding differences; best with one "
            "kernel per derivative set.  The flags below override the "
            "preset only when given."
        ),
    )
    parser.add_argument(
        "--optional-derivatives",
        action="store_const",
        const=True,
        default=None,
        help=(
            "declare every output as a Fortran 'optional' argument and "
            "compute it only when the caller passes it.  The generated "
            "subroutine can then be called with any subset of the "
            "derivatives."
        ),
    )
    parser.add_argument(
        "--no-array-cse",
        dest="array_cse",
        action="store_const",
        const=False,
        default=None,
        help=(
            "do not materialise index-dependent sub-expressions shared by "
            "several derivatives; keep the fully inlined expressions."
        ),
    )
    parser.add_argument(
        "--no-share",
        dest="share",
        action="store_const",
        const=False,
        default=None,
        help=(
            "do not compute shared summations and common scalar "
            "sub-expressions a single time; every output is emitted from "
            "its own fully inlined expressions (benchmark variant "
            "'V1 none' together with --no-array-cse)."
        ),
    )
    parser.add_argument(
        "--no-share-sums",
        dest="share_sums",
        action="store_const",
        const=False,
        default=None,
        help="disable only the shared-summation pass (ablation).",
    )
    parser.add_argument(
        "--no-scalar-cse",
        dest="scalar_cse",
        action="store_const",
        const=False,
        default=None,
        help="disable only the scalar common sub-expression pass (ablation).",
    )
    parser.add_argument(
        "--param-symmetry",
        action="store_const",
        const=True,
        default=None,
        help=(
            "exploit the parameters declared 'symmetric': canonicalise "
            "their subscripts and fold symmetric double sums into "
            "triangular loops."
        ),
    )
    parser.add_argument(
        "--hessian-symmetry",
        action="store_const",
        const=True,
        default=None,
        help="compute only the upper triangle of dfdndn and mirror it.",
    )
    parser.add_argument(
        "--sum-alpha",
        action="store_const",
        const=True,
        default=None,
        help="share summations equal up to renaming of their bound indices.",
    )
    parser.add_argument(
        "--hoist-invariants",
        action="store_const",
        const=True,
        default=None,
        help="move costly loop-invariant sub-expressions out of summations.",
    )
    parser.add_argument(
        "--cse-backend",
        choices=["thermodiff", "sympy"],
        default=None,
        help="algorithm of the scalar common sub-expression pass.",
    )
    parser.add_argument(
        "--work-array",
        action="store_const",
        const=True,
        default=None,
        help=(
            "store the array temporaries in one work array (one allocation "
            "per call instead of one per temporary)."
        ),
    )
    parser.add_argument(
        "--max-nc",
        type=int,
        default=None,
        help=(
            "give the array temporaries the compile-time extent MAX_NC "
            "(stack storage); the kernel stops with 'error stop' when nc "
            "exceeds it, so callers with larger nc need another kernel."
        ),
    )
    parser.add_argument(
        "--real-kind",
        default=None,
        help=(
            "kind of every real: an integer (default 8) or the name of a "
            "kind constant such as 'pr' (literals become 1.0_pr)."
        ),
    )
    parser.add_argument(
        "--kind-module",
        default=None,
        help=(
            "module that exports the --real-kind constant; adds "
            "'use <module>, only: <kind>' to the subroutine."
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
            preset=args.preset,
            workloads=args.workload,
            fixed_nc=args.fixed_nc,
            optional_derivatives=args.optional_derivatives,
            array_cse=args.array_cse,
            share=args.share,
            share_sums=args.share_sums,
            scalar_cse=args.scalar_cse,
            param_symmetry=args.param_symmetry,
            hessian_symmetry=args.hessian_symmetry,
            cse_backend=args.cse_backend,
            sum_alpha=args.sum_alpha,
            hoist_invariants=args.hoist_invariants,
            work_array=args.work_array,
            max_nc=args.max_nc,
            real_kind=args.real_kind,
            kind_module=args.kind_module,
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
