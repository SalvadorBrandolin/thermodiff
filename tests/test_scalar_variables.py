"""Derivatives with respect to extra scalar variables (``variables=``).

A cubic equation of state written with the scalars ``B``, ``D``, ``D1``
(mixture level covolume, attraction and a volume parameter) and the total
moles ``nT`` is differentiated with respect to them; the kernels are
compiled and compared with SymPy, and the chain rule over the composition
is assembled in NumPy and compared with the full-composition kernel.
"""

import itertools
import warnings

from fortran_tools import GFORTRAN, compile_and_run

import numpy as np

import pytest

import sympy as sp

from thermodiff import (
    DiffPlz,
    workload_module,
    T,
    V,
    idx_function,
    k,
    l,
    n,
    sum_components,
    to_fortran,
)
from thermodiff.codegen.options import CodegenOptions
from thermodiff.codegen.resolve import expression_argument_placeholders
from thermodiff.derivatives import resolve_derivatives

FLAGS = ("-O1", "-ffree-line-length-0")
PRESETS = ["minimal", "standard", "fastest"]
RELATIVE_TOLERANCE = 1e-12
SQRT2 = sp.sqrt(2)

B, D, D1, NT = sp.symbols("B D D1 nT")
SCALARS = (B, D, D1, NT)

#: Differentiation variables in the canonical order of the names.
VARIABLE_NAMES = ("B", "D", "D1", "nT", "T", "V")
SYMBOLS = {str(symbol): symbol for symbol in (*SCALARS, T, V)}


def derivative_name(factors):
    """Return the friendly name of the derivative with respect to factors."""
    if len(factors) == 2 and factors[0] == factors[1]:
        return f"d{factors[0]}2"
    return "".join("d" + factor for factor in factors)


#: Friendly name -> variables of every first and second derivative.
ALL_DERIVATIVES = {
    derivative_name(factors): factors
    for order in (1, 2)
    for factors in itertools.combinations_with_replacement(
        VARIABLE_NAMES, order
    )
}
POINT = {"V": 2.3, "T": 1.7, "B": 0.31, "D": 0.8, "D1": 1.3, "nT": 2.9}


def cubic_core(total_moles, covolume, attraction, volume_parameter):
    """Return the cubic residual Helmholtz energy in the given scalars."""
    attractive = (
        attraction
        / (2 * SQRT2 * volume_parameter * T)
        * sp.log((V + (1 + SQRT2) * covolume) / (V + (1 - SQRT2) * covolume))
    )
    return -total_moles * sp.log(1 - covolume / V) - attractive


def scalar_core_diff():
    """Return the ``DiffPlz`` of the scalar core, ``F(B, D, D1, nT, V, T)``."""
    expression = cubic_core(NT, B, D, D1)
    return DiffPlz(expression, indexes=[k], name="f", variables=list(SCALARS))


def module_text(source):
    """Wrap generated subroutines in a module."""
    return "module kern\ncontains\n" + source + "end module kern\n"


def real_literal(value):
    """Return a double precision Fortran literal."""
    return format(value, ".17e").replace("e", "d")


def declaration(name, value):
    """Return the declaration and the assignment of one driver input."""
    if isinstance(value, int):
        return f"integer :: {name}", f"{name} = {value}"
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        literal = real_literal(float(array))
        return f"real(8) :: {name}", f"{name} = {literal}"
    shape = ",".join(str(size) for size in array.shape)
    flat = ", ".join(real_literal(x) for x in array.flatten(order="F"))
    return (
        f"real(8) :: {name}({shape})",
        f"{name} = reshape([{flat}], [{shape}])",
    )


def run_kernel(
    source, sub_name, inputs, outputs, tmp_path, flags=FLAGS, wrap=True
):
    """Compile the kernel and a driver, run it and return the outputs.

    Parameters
    ----------
    source : str
        Generated subroutine text.
    sub_name : str
        Name of the subroutine.
    inputs : dict
        ``name -> int, float or array``, passed by keyword.
    outputs : dict
        ``name -> shape`` of every output to read.
    tmp_path : pathlib.Path
        Working directory.
    flags : tuple of str, optional
        Compiler flags.
    wrap : bool, optional
        Wrap ``source`` in the module ``kern``; ``False`` if it already
        is a module named ``kern``.

    Returns
    -------
    dict
        ``name -> float or array`` with the values of the outputs.
    """
    lines = ["program driver", "use kern", "implicit none"]
    declared = [declaration(name, value) for name, value in inputs.items()]
    lines.extend(line for line, _ in declared)
    for name, shape in outputs.items():
        extent = ",".join(str(size) for size in shape)
        lines.append(f"real(8) :: {name}" + (f"({extent})" if shape else ""))
    lines.extend(line for _, line in declared)
    arguments = ", &\n  ".join(
        f"{name}={name}" for name in list(inputs) + list(outputs)
    )
    lines.append(f"call {sub_name}({arguments})")
    lines.extend(f"write(*,'(*(ES26.17E3,1X))') {name}" for name in outputs)
    lines.append("end program driver")
    tmp_path.mkdir(parents=True, exist_ok=True)
    kernel_text = module_text(source) if wrap else source
    (tmp_path / "kern.f90").write_text(kernel_text)
    (tmp_path / "driver.f90").write_text("\n".join(lines) + "\n")
    result = compile_and_run(
        [tmp_path / "kern.f90", tmp_path / "driver.f90"], tmp_path, flags
    )
    values = {}
    for (name, shape), line in zip(
        outputs.items(), result.stdout.splitlines()
    ):
        numbers = np.array([float(word) for word in line.split()])
        values[name] = (
            numbers.reshape(shape, order="F") if shape else numbers[0]
        )
    return values


def generate(diff, derivatives, preset, optional=False):
    """Return the kernel of ``diff`` for the given derivatives."""
    options = CodegenOptions.from_preset(preset, optional_derivatives=optional)
    return to_fortran(diff, derivatives=derivatives, options=options)


@pytest.mark.skipif(GFORTRAN is None, reason="gfortran is not available")
class TestScalarCoreKernels:
    """First and second derivatives of ``F(B, D, D1, nT, V, T)``."""

    @pytest.mark.parametrize("optional", [False, True])
    @pytest.mark.parametrize("preset", PRESETS)
    def test_matches_sympy(self, preset, optional, tmp_path):
        diff = scalar_core_diff()
        names = list(ALL_DERIVATIVES)
        source = generate(diff, names, preset, optional)
        outputs = {"f": ()}
        outputs.update({"df" + name: () for name in names})
        inputs = dict(POINT)
        values = run_kernel(source, "f_derivs", inputs, outputs, tmp_path)

        expression = cubic_core(NT, B, D, D1)
        point = {SYMBOLS[name]: value for name, value in POINT.items()}
        assert values["f"] == pytest.approx(
            float(expression.subs(point)), rel=RELATIVE_TOLERANCE
        )
        for name, factors in ALL_DERIVATIVES.items():
            reference = sp.diff(expression, *(SYMBOLS[f] for f in factors))
            assert values["df" + name] == pytest.approx(
                float(reference.subs(point)), rel=RELATIVE_TOLERANCE
            ), name


CHECKED_FLAGS = ("-O1", "-ffree-line-length-0", "-fcheck=all")
#: Output sets served by a fixed kernel, by the fallback, and the full set.
WORKLOADS = ("dB,dBdD,dBdT", "dT,dV")
SUBSETS = (["dB", "dBdD", "dBdT"], ["dT", "dV"], ["dD", "dnT2", "dBdV"])


@pytest.mark.skipif(GFORTRAN is None, reason="gfortran is not available")
@pytest.mark.parametrize("preset", PRESETS)
def test_workload_dispatcher_with_scalar_variables(preset, tmp_path):
    """Any output subset through the dispatcher equals the full kernel."""
    diff = scalar_core_diff()
    source = workload_module(
        diff,
        list(WORKLOADS),
        derivatives=list(ALL_DERIVATIVES),
        module_name="kern",
        preset=preset,
        subroutine_name="f_derivs",
    )
    names = list(ALL_DERIVATIVES)
    full_outputs = {"f": ()} | {"df" + name: () for name in names}
    full = run_kernel(
        source,
        "f_derivs",
        POINT,
        full_outputs,
        tmp_path / "full",
        CHECKED_FLAGS,
        wrap=False,
    )
    for subset in SUBSETS:
        outputs = {"df" + name: () for name in subset}
        directory = tmp_path / "-".join(subset)
        values = run_kernel(
            source,
            "f_derivs",
            POINT,
            outputs,
            directory,
            CHECKED_FLAGS,
            wrap=False,
        )
        for name in outputs:
            assert values[name] == pytest.approx(
                full[name], rel=RELATIVE_TOLERANCE
            ), (subset, name)
    expression = cubic_core(NT, B, D, D1)
    point = {SYMBOLS[name]: value for name, value in POINT.items()}
    assert full["dfdBdT"] == pytest.approx(
        float(sp.diff(expression, B, T).subs(point)), rel=1e-11
    )


COVOLUME = sp.IndexedBase("b")
ATTRACTION = sp.IndexedBase("a")
VOLUME_PARAMETER = sp.IndexedBase("c")
NC = 3
COMPOSITION = np.array([0.7, 1.1, 1.3])
COVOLUMES = np.array([0.11, 0.17, 0.23])
VOLUME_PARAMETERS = np.array([1.2, 0.9, 1.5])
ATTRACTIONS = np.array([[0.9, 0.5, 0.6], [0.5, 1.4, 0.7], [0.6, 0.7, 1.1]])
FULL_DERIVATIVES = ["dV", "dT", "dn", "dndn", "dndT", "dndV"]


def full_composition_diff():
    """Return the cubic with mixing rules written in the mole numbers."""
    total = sum_components(n[k], k)
    covolume = sum_components(COVOLUME[k] * n[k], k)
    attraction = sum_components(
        sum_components(ATTRACTION[k, l] * n[k] * n[l], l), k
    )
    volume_parameter = sum_components(VOLUME_PARAMETER[k] * n[k], k)
    expression = cubic_core(total, covolume, attraction, volume_parameter)
    return DiffPlz(expression, indexes=[k, l], name="ar")


def assemble_from_scalar_core(core):
    """Apply the chain rule over the composition to the scalar derivatives.

    ``core`` maps the names of the scalar kernel outputs (``dfdB`` ...) to
    their values; the mixing rules are ``B = b.n``, ``D = n.a.n``,
    ``D1 = c.n`` and ``nT = sum(n)``.
    """
    gradients = {
        "B": COVOLUMES,
        "D": 2 * ATTRACTIONS @ COMPOSITION,
        "D1": VOLUME_PARAMETERS,
        "nT": np.ones(NC),
    }
    hessians = {"D": 2 * ATTRACTIONS}
    names = list(gradients)

    def second(first, second_name):
        key = "df" + derivative_name(
            tuple(sorted((first, second_name), key=VARIABLE_NAMES.index))
        )
        return core[key]

    gradient = sum(core["dfd" + x] * gradients[x] for x in names)
    hessian = (
        sum(
            second(x, y) * np.outer(gradients[x], gradients[y])
            for x in names
            for y in names
        )
        + core["dfdD"] * hessians["D"]
    )
    mixed = {
        thermo: sum(core[f"dfd{x}d{thermo}"] * gradients[x] for x in names)
        for thermo in ("T", "V")
    }
    return gradient, hessian, mixed


@pytest.mark.skipif(GFORTRAN is None, reason="gfortran is not available")
@pytest.mark.parametrize("preset", PRESETS)
def test_scalar_core_assembles_the_full_composition_kernel(preset, tmp_path):
    """The scalar core plus the chain rule equals the full kernel."""
    full = run_kernel(
        generate(full_composition_diff(), FULL_DERIVATIVES, preset),
        "ar_derivs",
        {
            "nc": NC,
            "V": POINT["V"],
            "T": POINT["T"],
            "n": COMPOSITION,
            "b": COVOLUMES,
            "a": ATTRACTIONS,
            "c": VOLUME_PARAMETERS,
        },
        {
            "ar": (),
            "dardV": (),
            "dardT": (),
            "dardn": (NC,),
            "dardndn": (NC, NC),
            "dardndT": (NC,),
            "dardndV": (NC,),
        },
        tmp_path,
    )
    names = [
        name
        for name, factors in ALL_DERIVATIVES.items()
        if not set(factors) <= {"T", "V"}
    ] + ["dT", "dV"]
    point = {
        "B": COVOLUMES @ COMPOSITION,
        "D": COMPOSITION @ ATTRACTIONS @ COMPOSITION,
        "D1": VOLUME_PARAMETERS @ COMPOSITION,
        "nT": COMPOSITION.sum(),
        "V": POINT["V"],
        "T": POINT["T"],
    }
    outputs = {"f": ()} | {"df" + name: () for name in names}
    core = run_kernel(
        generate(scalar_core_diff(), names, preset),
        "f_derivs",
        point,
        outputs,
        tmp_path,
    )
    gradient, hessian, mixed = assemble_from_scalar_core(core)
    assert core["f"] == pytest.approx(full["ar"], rel=RELATIVE_TOLERANCE)
    assert gradient == pytest.approx(full["dardn"], rel=1e-11)
    assert hessian == pytest.approx(full["dardndn"], rel=1e-11)
    assert mixed["T"] == pytest.approx(full["dardndT"], rel=1e-11)
    assert mixed["V"] == pytest.approx(full["dardndV"], rel=1e-11)
    assert core["dfdT"] == pytest.approx(full["dardT"], rel=1e-11)
    assert core["dfdV"] == pytest.approx(full["dardV"], rel=1e-11)


class TestVocabulary:
    """Names, groups and the default behaviour."""

    variables = ["B", "D", "D1"]

    def test_names_are_canonical(self):
        names = ["dB", "dDdB", "dTdB", "dfdB2", "dndB", "dBdn", "dD1"]
        selected = resolve_derivatives(names, scalar_variables=self.variables)
        assert selected == [
            "var_dB",
            "var_dD1",
            "var_dndB",
            "var_dB2",
            "var_dBdD",
            "var_dBdT",
        ]

    def test_groups_include_the_scalar_derivatives(self):
        first = resolve_derivatives("first", scalar_variables=self.variables)
        second = resolve_derivatives("second", scalar_variables=self.variables)
        assert first[-3:] == ["var_dB", "var_dD", "var_dD1"]
        assert "var_dBdD1" in second and "var_dndD" in second
        assert not set(first) & set(second)

    def test_groups_respect_the_declared_variables(self):
        selected = resolve_derivatives(
            "all",
            declared_variables=["T", "n"],
            scalar_variables=self.variables,
        )
        assert "var_dBdT" in selected and "var_dBdV" not in selected
        with pytest.raises(ValueError, match="requires the variable"):
            resolve_derivatives(
                "dBdV",
                declared_variables=["T", "n"],
                scalar_variables=self.variables,
            )

    def test_unknown_variable_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown derivatives: dX"):
            resolve_derivatives("dX", scalar_variables=self.variables)
        with pytest.raises(ValueError, match="Unknown derivatives: dB"):
            resolve_derivatives("dB")

    def test_no_variables_changes_nothing(self):
        diff = DiffPlz(NT * sp.log(V) / T, indexes=[k], name="f")
        assert diff.derivatives is None
        assert diff.scalar_derivatives == {}
        assert diff.variables == []

    def test_all_are_computed_without_a_selection(self):
        diff = scalar_core_diff()
        assert len(diff.scalar_derivatives) == 4 + 4 + 10 + 4 * 3
        assert "var_dBdD" in diff.derivatives

    @pytest.mark.parametrize("bad", ["T", "v", "n", "N"])
    def test_reserved_names_are_rejected(self, bad):
        with pytest.raises(ValueError, match="thermodynamic variable"):
            DiffPlz(T, variables=[sp.Symbol(bad)])

    def test_case_clash_message_explains_and_suggests(self):
        with pytest.raises(ValueError) as caught:
            DiffPlz(T, variables=[sp.Symbol("N")])
        message = str(caught.value)
        assert "'N' clashes with the thermodynamic variable 'n'" in message
        assert "case-insensitive" in message
        assert "choose another name" in message

    def test_variables_must_be_plain_symbols(self):
        with pytest.raises(ValueError, match="plain symbols"):
            DiffPlz(T, variables=[n[k]])

    def test_values_match_sympy(self):
        diff = DiffPlz(
            NT * B * sp.log(V) * D**2 / T,
            name="f",
            variables=[B, D, NT],
            derivatives=["dBdD", "dnTdB", "dBdT"],
        )
        assert diff.var_dBdD == sp.diff(diff.expression, B, D)
        assert diff.var_dBdnT == sp.diff(diff.expression, B, NT)
        assert diff.var_dBdT == sp.diff(diff.expression, B, T)
        assert diff.dt is None and diff.var_dB is None


class TestEmission:
    """Order, names and declarations of the generated outputs."""

    def test_outputs_follow_the_canonical_order(self):
        diff = scalar_core_diff()
        names = ["dnT", "dBdT", "dB2", "dD", "dBdD", "dT", "dB", "dV"]
        source = generate(diff, names, "minimal")
        signature = source.splitlines()[0]
        assert signature.endswith(
            "f, dfdT, dfdV, dfdB, dfdD, dfdnT, dfdB2, dfdBdD, dfdBdT)"
        )

    def test_variables_are_real_inputs(self):
        source = generate(scalar_core_diff(), ["dB"], "standard")
        for name in ("B", "D", "D1", "nT"):
            assert f"real(8), intent(in) :: {name}" in source

    def test_a_derivative_without_a_selection_is_not_emitted_if_zero(self):
        diff = DiffPlz(B * V, name="f", variables=[B, D])
        source = to_fortran(diff)
        assert "dfdB" in source and "dfdD" not in source

    def test_array_valued_mixed_derivative(self):
        weights = sp.IndexedBase("w")
        expression = B * sum_components(weights[k] * n[k], k) * V
        diff = DiffPlz(expression, indexes=[k], name="f", variables=[B])
        source = to_fortran(diff, derivatives=["dndB", "dB", "dn"])
        assert "real(8), intent(out) :: dfdndB(:)" in source
        assert "real(8), intent(out) :: dfdB\n" in source

    def test_missing_selection_reports_how_to_rebuild(self):
        diff = DiffPlz(B * V, name="f", variables=[B], derivatives=["dV"])
        with pytest.raises(ValueError, match="not computed"):
            to_fortran(diff, derivatives=["dB"])


class TestExpressionArguments:
    """A placeholder fed with expressions warns that totals are needed."""

    @staticmethod
    def alpha_model(argument):
        alpha = idx_function("alpha")(k, argument)
        expression = sum_components(n[k] * alpha, k)
        return DiffPlz(expression, internal_functions=[alpha], indexes=[k])

    @pytest.mark.parametrize("preset", PRESETS)
    def test_expression_argument_warns_once(self, preset):
        total = sum_components(n[l], l)
        diff = self.alpha_model(total / V)
        options = CodegenOptions.from_preset(preset)
        with pytest.warns(UserWarning) as caught:
            to_fortran(diff, derivatives=["dV", "dn"], options=options)
        messages = [
            str(w.message)
            for w in caught
            if str(w.message).startswith("placeholder ")
        ]
        assert len(messages) == 1
        assert "'alpha'" in messages[0]
        assert "total derivatives" in messages[0]
        assert "/V" in messages[0]

    def test_fastest_generates_the_same_inputs_as_standard(self):
        """``nested_nodes`` used to fail differentiating w.r.t. a Sum."""
        diff = self.alpha_model(sum_components(n[l], l) / V)
        signatures = {}
        for preset in ("standard", "fastest"):
            options = CodegenOptions.from_preset(preset)
            with pytest.warns(UserWarning):
                text = to_fortran(
                    diff, derivatives=["dV", "dn"], options=options
                )
            signatures[preset] = text.split("&\n")[0].split("(", 1)[1]
        assert signatures["fastest"].split(", ")[:5] == (
            signatures["standard"].split(", ")[:5]
        )

    def test_fastest_reports_that_nested_nodes_is_disabled(self):
        diff = self.alpha_model(sum_components(n[l], l) / V)
        options = CodegenOptions.from_preset("fastest")
        with pytest.warns(UserWarning, match="nested_nodes is disabled"):
            to_fortran(diff, derivatives=["dV", "dn"], options=options)

    def test_variable_arguments_do_not_warn(self):
        diff = self.alpha_model(V)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            to_fortran(diff, derivatives=["dV"])


@pytest.mark.parametrize("preset", ["minimal", "standard", "fastest"])
def test_vector_placeholder_arguments_do_not_warn(preset):
    """``B(n)`` and ``D(n, T)`` take variables, not expressions."""
    b_func = idx_function("B")(n)
    d_func = idx_function("D")(n, T)
    diff = DiffPlz(
        sum_components(n[l], l) * sp.log(1 - b_func / V) + d_func / T,
        internal_functions=[b_func, d_func],
        indexes=[l],
        name="f",
    )
    assert expression_argument_placeholders(diff.expression) == []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        to_fortran(
            diff,
            derivatives=["dn", "dndn", "dT", "dndT"],
            options=CodegenOptions.from_preset(preset),
        )
