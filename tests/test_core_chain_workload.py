"""Fused scalar-chain workload kernels (``scalar_chain_workload_module``).

The numeric tests compile the generated module with gfortran and compare,
for every workload and for output sets that are not workloads (the
fallback), the dispatcher with the separate path: the core kernel followed
by ``scalar_chain_routine``.  Two scalar cores are used: the generic cubic
equation of state (``total_moles`` and a ``T``-dependent scalar) and a core
with ``linear`` scalars, scalars that depend on ``V`` and structurally zero
derivatives.
"""

import re

from fortran_tools import GFORTRAN, compile_and_run

import numpy as np

import pytest

import sympy as sp

import thermodiff as td
from thermodiff.codegen.core_chain import chain_fragments, resolve_chain
from thermodiff.codegen.core_chain_workload import (
    fuse_kernel,
    scalar_chain_workload_module,
)
from thermodiff.codegen.options import CodegenOptions
from thermodiff.codegen.workload import wrap_module

FLAGS = ("-O1", "-ffree-line-length-0", "-fcheck=all")
TOLERANCE = 1e-13
SEED = 20260101
GAS_CONSTANT = 0.0831446261815324
SIZES = (2, 5, 20)
PRESET = "standard"

B, D, E, S = sp.symbols("B D E S")
X1, X2, X3 = sp.symbols("X1 X2 X3")
R = sp.Symbol("R")


def cubic_core():
    """Return the generic cubic core as a scalar core."""
    e2 = (1 - E) / (1 + E)
    expression = -S * R * td.T * sp.log(1 - B / td.V) - D * sp.log(
        (td.V + E * B) / (td.V + e2 * B)
    ) / (B * (E - e2))
    return td.DiffPlz(expression, name="f", variables=[B, D, E, S])


def mixed_core():
    """Return a core with linear, V dependent and zero-derivative scalars."""
    expression = (
        S * sp.log(1 + X1 / td.V)
        + X2 * X3 / (1 + X1)
        - X1 * X2 / (td.T * td.V)
        + sp.sqrt(S) * X3 * td.T
    )
    return td.DiffPlz(expression, name="f", variables=[X1, X2, X3, S])


CASES = {
    "cubic": dict(
        build=cubic_core,
        scalars=["B", "D", "E", "S"],
        declarations=dict(depends_on={"D": ["T"]}, total_moles=("S",)),
        symbols=["R"],
        workloads=[["dv", "dv2"], ["dv", "dni"], ["dt", "dv", "dni"]],
        others=[["dt", "dni"], ["dnidnj"], "all"],
    ),
    "mixed": dict(
        build=mixed_core,
        scalars=["X1", "X2", "X3", "S"],
        declarations=dict(
            depends_on={"X2": ["T"], "X3": ["V"]},
            linear=("X1",),
            total_moles=("S",),
        ),
        symbols=[],
        workloads=[["dv"], ["dni", "dnidnj"], ["dt", "dtdv", "dvdni"]],
        others=[["dt", "dv", "dni"], ["dtdni", "dvdni"], "all"],
    ),
}


def generate(case, workloads=None, **overrides):
    """Return the fused module of ``case``."""
    spec = CASES[case]
    return scalar_chain_workload_module(
        spec["build"](),
        spec["scalars"],
        spec["workloads"] if workloads is None else workloads,
        spec["declarations"].get("depends_on"),
        **{k: v for k, v in spec["declarations"].items() if k != "depends_on"},
        module_symbols=spec["symbols"],
        uses=["consts, only: R"],
        preset=PRESET,
        subroutine_name="fused",
        **overrides,
    )


def pieces_of(case, selection, optional=False):
    """Return the chain fragments of an output selection."""
    spec = CASES[case]
    model, requested = resolve_chain(
        spec["scalars"],
        spec["declarations"].get("depends_on"),
        spec["declarations"].get("linear", ()),
        spec["declarations"].get("total_moles", ()),
        spec["build"](),
        None,
        ("T", "V"),
        selection,
        "f",
        "ar",
    )
    return chain_fragments(model, requested, optional)


def split_declaration(line):
    """Return ``(name, rank, is_output)`` of a chain dummy declaration."""
    declared = line.split("::")[1]
    name = declared.split("(")[0].strip()
    rank = declared.count(",") + 1 if "(" in declared else 0
    return name, rank, "intent(out)" in line


def reference_sources(case, selection, index):
    """Return the separate core and chain routines of ``selection``."""
    spec = CASES[case]
    declarations = dict(spec["declarations"])
    chain_args = (spec["scalars"], declarations.pop("depends_on", None))
    core = spec["build"]()
    names = td.scalar_chain_core_derivatives(
        *chain_args, core=core, derivatives=selection, **declarations
    )
    options = CodegenOptions.from_preset(PRESET)
    core_text = td.to_fortran(
        core,
        derivatives=names,
        subroutine_name=f"core_{index}",
        module_symbols=spec["symbols"],
        options=options,
    )
    chain_text = td.scalar_chain_routine(
        *chain_args,
        core=core,
        derivatives=selection,
        subroutine_name=f"chain_{index}",
        **declarations,
    )
    return core_text, chain_text


def fortran_real(value):
    """Return a double precision Fortran literal."""
    return repr(float(value)).replace("e", "d") + (
        "d0" if "e" not in repr(float(value)) else ""
    )


def fortran_value(array):
    """Return a Fortran expression for a NumPy array (column-major)."""
    flat = ", ".join(fortran_real(x) for x in array.ravel(order="F"))
    if array.ndim == 0:
        return flat
    shape = ", ".join(str(extent) for extent in array.shape)
    return f"reshape([{flat}], [{shape}])"


class Program:
    """Accumulates the declarations and statements of the driver."""

    def __init__(self, case, nc):
        self.case = case
        self.nc = nc
        self.rng = np.random.default_rng(SEED + nc)
        self.declared = {}
        self.initial = []
        self.compare = []

    def declare(self, name, rank, random=False):
        """Declare a variable once; give inputs random values."""
        if name in self.declared:
            return
        self.declared[name] = rank
        if random:
            values = self.rng.uniform(-1.0, 1.0, (self.nc,) * rank)
            self.initial.append(f"   {name} = {fortran_value(values)}")

    def text(self):
        """Return the program."""
        scalars = CASES[self.case]["scalars"]
        lines = [
            "program drive",
            "   use consts, only: R",
            "   use fused_mod",
            "   use reference_mod",
            "   implicit none",
            f"   integer, parameter :: nc = {self.nc}",
            "   integer :: unit",
            f"   real(8) :: V = 2.3d0, T = 1.7d0, f_ref, f_new, "
            f"{', '.join(scalars)}",
        ]
        for name, rank in self.declared.items():
            shape = f"({', '.join(['nc'] * rank)})" if rank else ""
            lines.append(f"   real(8) :: {name}{shape}")
        for name in scalars:
            value = self.rng.uniform(0.3, 1.1)
            self.initial.append(f"   {name} = {fortran_real(value)}")
        return "\n".join(
            [
                *lines,
                "   open(newunit=unit, file='out.txt')",
                *self.initial,
                *self.compare,
                "   close(unit)",
                "end program",
            ]
        )


def add_comparison(program, selection, position):
    """Add the statements that run and compare one output set."""
    pieces = pieces_of(program.case, selection)
    core_names = list(pieces.inputs.core)
    inputs, outputs = [], []
    for line in pieces.declarations[1:]:
        name, rank, is_output = split_declaration(line)
        if name not in core_names:
            (outputs if is_output else inputs).append((name, rank))
    for name in core_names:
        program.declare(name, 0)
    for name, rank in inputs:
        program.declare(name, rank, random=True)
    for name, rank in outputs:
        program.declare(f"ref_{name}", rank)
        program.declare(f"new_{name}", rank)
    scalars = CASES[program.case]["scalars"]
    point = [f"{a}={a}" for a in ["V", "T", *scalars]]
    given = [f"{name}={name}" for name, _ in inputs]
    chain = ["nc=nc"] + [f"{n}={n}" for n in core_names] + given
    chain += [f"{n}=ref_{n}" for n, _ in outputs]
    fused = ["nc=nc", *point, "f=f_new", *given]
    fused += [f"{n}=new_{n}" for n, _ in outputs]
    core = [*point, "f=f_ref", *(f"{n}={n}" for n in core_names)]
    program.compare += [
        f"   call core_{position}({', '.join(core)})",
        f"   call chain_{position}({', '.join(chain)})",
        f"   call fused({', '.join(fused)})",
        f"   write(unit, *) {position}, 'f', "
        "abs(f_new - f_ref)/max(abs(f_ref), tiny(1d0))",
    ]
    for name, _ in outputs:
        program.compare.append(
            f"   write(unit, *) {position}, '{name}', "
            f"maxval([abs(new_{name} - ref_{name})])/"
            f"max(maxval([abs(ref_{name})]), tiny(1d0))"
        )


def run_case(case, selections, nc, tmp_path, workloads=None):
    """Compile and run; return the relative errors by (set, output)."""
    consts = (
        "module consts\n   implicit none\n"
        f"   real(8), parameter :: R = {GAS_CONSTANT}d0\nend module\n"
    )
    reference = wrap_module(
        "reference_mod",
        ["consts, only: R"],
        "\n\n".join(
            "\n\n".join(reference_sources(case, selection, index))
            for index, selection in enumerate(selections)
        ),
    )
    program = Program(case, nc)
    for position, selection in enumerate(selections):
        add_comparison(program, selection, position)
    program = program.text()
    sources = []
    for name, text in (
        ("consts", consts),
        ("fused", generate(case, workloads)),
        ("reference", reference),
        ("drive", program),
    ):
        path = tmp_path / f"{name}.f90"
        path.write_text(text)
        sources.append(path)
    result = compile_and_run(sources, tmp_path, FLAGS)
    assert result.returncode == 0
    errors = {}
    for line in (tmp_path / "out.txt").read_text().splitlines():
        position, name, error = line.split()
        errors[int(position), name] = float(error)
    return errors


def output_sets(case):
    """Return the selections to compare: workloads, then other sets."""
    spec = CASES[case]
    return [*spec["workloads"], *spec["others"]]


@pytest.mark.slow
@pytest.mark.skipif(GFORTRAN is None, reason="gfortran not available")
@pytest.mark.parametrize("nc", SIZES)
@pytest.mark.parametrize("case", sorted(CASES))
def test_dispatcher_matches_separate_core_and_chain(case, nc, tmp_path):
    selections = output_sets(case)
    errors = run_case(case, selections, nc, tmp_path)
    assert {position for position, _ in errors} == set(range(len(selections)))
    worst = max(errors.values())
    assert worst <= TOLERANCE, errors


def procedure_texts(source):
    """Return ``{name: text}`` of the subroutines of a module."""
    pattern = re.compile(
        r"^\s*subroutine (\w+)\(.*?^\s*end subroutine \1", re.M | re.S
    )
    return {m.group(1): m.group(0) for m in pattern.finditer(source)}


@pytest.mark.parametrize("case", sorted(CASES))
def test_workload_kernels_have_no_optional_arguments(case):
    source = generate(case)
    texts = procedure_texts(source)
    workloads = len(CASES[case]["workloads"])
    kernels = {k: v for k, v in texts.items() if re.search(r"_w\d+$", k)}
    assert len(kernels) == workloads
    for name, text in kernels.items():
        assert "optional" not in text and "present(" not in text, name
        assert "call " not in text, name
        assert "NOINLINE" in text
    assert source.count("   case (") == workloads
    assert "present(" in texts["fused_fallback"]


def test_workload_kernel_computes_only_the_core_derivatives_it_needs():
    texts = procedure_texts(generate("cubic"))
    volume = texts["fused_w1"]
    assert "dfdV" in volume and "dfdB" not in volume
    assert "dBdn" not in volume


def test_repeated_output_set_is_rejected():
    with pytest.raises(ValueError, match="repeats"):
        generate("cubic", [["dv"], ["dv"]])


def test_workload_outside_the_signature_is_rejected():
    with pytest.raises(ValueError):
        generate("cubic", [["dvdni"]], derivatives=["dv"])


def test_real_kind_is_applied_to_the_chain_statements():
    source = generate("cubic", real_kind="pr", kind_module="yaeos__constants")
    assert "real(8)" not in source
    assert "use yaeos__constants, only: pr" in source


def test_fuse_kernel_rejects_a_core_with_nc():
    core = td.to_fortran(
        td.DiffPlz(td.T * sum_nc(), name="f"),
        derivatives=["dT"],
    )
    pieces = pieces_of("mixed", ["dv"])
    with pytest.raises(ValueError, match="nc"):
        fuse_kernel(core, pieces, name="k", options=CodegenOptions())


def sum_nc():
    """Return an expression with a composition sum (a kernel with nc)."""
    index = td.k
    return td.sum_components(td.n[index], index)
