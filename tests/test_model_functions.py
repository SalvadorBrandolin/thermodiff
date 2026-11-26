"""Placeholders in the ``.model`` language: the ``functions`` section.

A ``functions`` declaration builds the same SymPy objects as the Python API
(``idx_function`` applied to index arguments and variables, passed as
``internal_functions``), so the generated kernel must equal the kernel of
the equivalent ``DiffPlz`` (issue ipqa-research/thermodiff#12).
"""

import pytest

import sympy as sp

from thermodiff import DiffPlz, T, V, idx_function, k, l, n, sum_components
from thermodiff.codegen.options import resolve_codegen_options
from thermodiff.derivatives import resolve_derivatives
from thermodiff.fortran import to_fortran
from thermodiff.model import ModelError, compile_model
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

PRESETS = ("minimal", "standard", "fastest")
ALPHA = sp.Float("0.3")

HEADER = """\
model {name}
variables
    n   nc
    V
    T
functions
{functions}
function
    {function}
"""


def model_text(name, functions, function):
    """Return a ``.model`` text with the given placeholders and function."""
    block = "\n".join("    " + line for line in functions)
    return HEADER.format(name=name, functions=block, function=function)


def api_kernel(diff, name, preset):
    """Return the kernel of ``diff`` as the ``.model`` emitter writes it."""
    derivatives = resolve_derivatives(["all"], {"n", "T", "V"})
    code = to_fortran(
        diff,
        derivatives=derivatives,
        name="f",
        subroutine_name=f"{name}_derivs",
        integer_args=[],
        module_symbols=[],
        symmetric_params=(),
        options=resolve_codegen_options(preset, None),
    )
    return code


def model_kernel(text, preset):
    """Return the subroutine text of a compiled ``.model``."""
    module = compile_model(text, preset=preset)
    body = module.split("contains\n", 1)[1].rsplit("end module", 1)[0]
    lines = [
        line[3:] if line.startswith("   ") else line
        for line in body.splitlines()
    ]
    return "\n".join(lines).strip()


def issue_b_diff():
    """``f = sum_l n_l log(1 - B(n)/V)`` with a placeholder ``B(n)``."""
    b_func = idx_function("B")(n)
    f = sum_components(n[k] * sp.log(1 - b_func / V), k)
    return DiffPlz(f, internal_functions=[b_func], indexes=[k], name="f")


def issue_d_diff():
    """``f = T sum_l n_l log(D(n, T))`` with a placeholder ``D(n, T)``."""
    d_func = idx_function("D")(n, T)
    f = T * sum_components(n[k] * sp.log(d_func), k)
    return DiffPlz(f, internal_functions=[d_func], indexes=[k], name="f")


def nrtl_diff():
    """NRTL-like excess energy with ``tau(l, k, T)`` weights."""
    tau = idx_function("tau")(l, k, T)
    gamma = sp.exp(-ALPHA * tau)
    numerator = sum_components(n[l] * tau * gamma, l)
    denominator = sum_components(n[l] * gamma, l)
    expr = sum_components(n[k] * numerator / denominator, k)
    return DiffPlz(expr, internal_functions=[tau], indexes=[k, l], name="f")


CASES = {
    "scalar_b": (
        ["B   (n)"],
        "sum_k n_k * log(1 - B/V)",
        issue_b_diff,
    ),
    "mixed_d": (
        ["D   (n, T)"],
        "T * sum_k n_k * log(D)",
        issue_d_diff,
    ),
    "nrtl": (
        ["tau  nc nc  (T)"],
        "sum_k n_k * (sum_l n_l tau_lk exp(-0.3 tau_lk))"
        " / (sum_l n_l exp(-0.3 tau_lk))",
        nrtl_diff,
    ),
}


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("case", sorted(CASES))
def test_kernel_equals_python_api(case, preset):
    functions, function, make_diff = CASES[case]
    text = model_text(case, functions, function)
    expected = api_kernel(make_diff(), case, preset)
    assert model_kernel(text, preset) == expected.strip()


class TestParsing:
    def test_declaration_fields(self):
        raw = read_sections(
            model_text("m", ["tau nc nc (T)", "B (n, V)"], "V")
        )
        tau, scalar = raw.functions
        assert (tau.name, tau.dims, tau.args) == ("tau", ("nc", "nc"), ("T",))
        assert (scalar.name, scalar.dims, scalar.args) == (
            "B",
            (),
            ("n", "V"),
        )

    def test_functions_is_not_function(self):
        raw = read_sections(model_text("m", ["B (n)"], "B"))
        assert raw.function.rhs == "B"
        assert [decl.name for decl in raw.functions] == ["B"]

    def test_symmetric_token_is_parsed(self):
        raw = read_sections(model_text("m", ["tau nc nc symmetric (T)"], "V"))
        assert raw.functions[0].attributes == ("symmetric",)

    def test_symmetric_placeholder_is_registered(self):
        text = model_text("m", ["a nc nc symmetric (T)"], "sum_k a_kk")
        ir = build_ir(read_sections(text))
        assert ir.placeholders[0].symmetric
        assert ir.symmetric == ["a"]

    def test_symmetric_needs_a_square_pair(self):
        text = model_text("m", ["b nc symmetric (T)"], "sum_k b_k")
        with pytest.raises(ModelError, match="two index arguments"):
            build_ir(read_sections(text))

    def test_positive_is_not_supported(self):
        text = model_text("m", ["a nc nc positive (T)"], "sum_k a_kk")
        with pytest.raises(ModelError, match="'positive' is not supported"):
            build_ir(read_sections(text))

    def test_subscripted_declaration_name(self):
        text = model_text(
            "m", ["tau_ij (T)"], "sum_k sum_l n_k * n_l * tau_kl"
        )
        assert build_ir(read_sections(text)).placeholders[0].rank == 2

    def test_unused_placeholder_warns(self):
        text = model_text("m", ["B (n)"], "T")
        ir = build_ir(read_sections(text))
        assert any("'B' is declared but never used" in w for w in ir.warnings)


def compile_error(functions, function="T", **kwargs):
    """Return the message of the error raised by a faulty model."""
    text = model_text("m", functions, function)
    with pytest.raises(ModelError) as caught:
        compile_model(text, **kwargs)
    return str(caught.value)


class TestErrors:
    def test_functions_after_function(self):
        text = "model m\nvariables\n  T\nfunction\n  T\nfunctions\n  B (T)\n"
        with pytest.raises(ModelError, match="must come before 'function'"):
            read_sections(text)

    def test_unknown_dimension(self):
        message = compile_error(["tau ng ng (T)"], "T")
        assert "unknown dimension 'ng'" in message

    def test_known_extra_dimension_is_accepted(self):
        text = (
            "model m\nvariables\n  n nc\n  T\nparameters\n  w ng\n"
            "functions\n  g ng (T)\nfunction\n  sum_a[1:ng] w_a * g_a\n"
        )
        assert build_ir(read_sections(text)).placeholders[0].dims == ("ng",)

    def test_argument_is_not_a_variable(self):
        message = compile_error(["B (x)"])
        assert "argument 'x' of placeholder 'B' is not a declared" in message

    def test_argument_not_differentiable(self):
        text = (
            "model m\nvariables\n  n nc\n  T\n  z\nfunctions\n  B (z)\n"
            "function\n  B\n"
        )
        with pytest.raises(ModelError, match="must be one of n, T, V, P"):
            build_ir(read_sections(text))

    def test_undeclared_pressure(self):
        assert "'P' of placeholder 'B' is not a declared" in compile_error(
            ["B (P)"]
        )

    def test_missing_arguments(self):
        assert "form 'name [dims] (args)'" in compile_error(["B"])

    def test_no_arguments(self):
        assert "at least one variable argument" in compile_error(["B ()"])

    def test_duplicate_name(self):
        assert "'T' is declared more than once" in compile_error(["T (n)"])

    def test_builtin_name(self):
        assert "'exp' is declared more than once" in compile_error(["exp (T)"])

    def test_wrong_subscript_count(self):
        message = compile_error(
            ["tau nc nc (T)"], "sum_k sum_l n_k * n_l * tau_k"
        )
        assert "expects 2 subscript(s), got 1" in message

    def test_missing_subscripts(self):
        message = compile_error(["tau nc nc (T)"], "tau")
        assert "'tau' requires 2 subscript(s)" in message


def symmetric_diff():
    """Return the Python-API ``DiffPlz`` of the symmetric quadratic rule."""
    a = idx_function("a", symmetric=(0, 1))(k, l, T)
    return DiffPlz(
        sum_components(sum_components(n[k] * n[l] * a, l), k),
        internal_functions=[a],
        indexes=[k, l],
        name="f",
    )


@pytest.mark.parametrize("preset", PRESETS)
def test_symmetric_kernel_equals_python_api(preset):
    """``symmetric`` gives the kernel of ``symmetric_params=["a"]`` (#14)."""
    text = model_text(
        "symq", ["a nc nc symmetric (T)"], "sum_k sum_l n_k * n_l * a_kl"
    )
    derivatives = resolve_derivatives(["all"], {"n", "T", "V"})
    expected = to_fortran(
        symmetric_diff(),
        derivatives=derivatives,
        name="f",
        subroutine_name="symq_derivs",
        integer_args=[],
        module_symbols=[],
        symmetric_params=("a",),
        options=resolve_codegen_options(preset, None),
    )
    assert model_kernel(text, preset) == expected.strip()
