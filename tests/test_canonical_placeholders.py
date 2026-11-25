"""Canonical names of placeholder derivatives (issue #8).

The gradient of a placeholder ``B(n)`` appears under two index names in a
Hessian.  Without ``canonical_placeholders`` it becomes two arguments
``dBdi`` and ``dBdj`` holding the same data; with it, one argument
``dBdn`` read as ``dBdn(i)`` and ``dBdn(j)``, so the CSE passes merge the
copies and the Hessian ``dBdndn`` is a symmetric parameter.
"""

import re
import warnings

import pytest

import sympy as sp

from thermodiff import DiffPlz, T, V, idx_function, l, n, sum_components
from thermodiff.codegen.options import CodegenOptions
from thermodiff.codegen.resolve import _canonical_suffix
from thermodiff.fortran import to_fortran

i, j = sp.Idx("i"), sp.Idx("j")


def signature(text):
    """Return the dummy arguments of the generated subroutine."""
    flat = re.sub(r"&\n\s*", "", text)
    head = re.search(r"subroutine \w+\(([^)]*)\)", flat).group(1)
    return [arg.strip() for arg in head.split(",")]


def repro_diff():
    """The #8 reproducer: ``f = n_T log(1 - B(n)/V)``."""
    b_func = idx_function("B")(n)
    f = sum_components(n[l], l) * sp.log(1 - b_func / V)
    return DiffPlz(f, internal_functions=[b_func], indexes=[l], name="f")


def mixed_diff():
    """A placeholder of ``n`` and ``T``: gives the mixed ``d2D/dT dn``."""
    d_func = idx_function("D")(n, T)
    f = sum_components(n[l], l) * sp.log(d_func) * T
    return DiffPlz(f, internal_functions=[d_func], indexes=[l], name="f")


def generate(diff, derivatives, preset="fastest", **extra):
    options = CodegenOptions.from_preset(preset, **extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return to_fortran(diff, derivatives=derivatives, options=options)


class TestSuffix:
    def test_gradient(self):
        assert _canonical_suffix((n[i],)) == ("dn", ["i"])

    def test_hessian_keeps_the_index_order(self):
        assert _canonical_suffix((n[j], n[i])) == ("dndn", ["j", "i"])

    @pytest.mark.parametrize("variables", [(T, n[i]), (n[i], T)])
    def test_mixed_puts_scalars_first(self, variables):
        assert _canonical_suffix(variables) == ("dTdn", ["i"])

    def test_scalars_are_sorted(self):
        assert _canonical_suffix((V, T, T)) == ("dTdTdV", [])


class TestReproducer:
    def test_option_default(self):
        assert CodegenOptions().canonical_placeholders is True

    def test_index_names_when_the_option_is_off(self):
        text = generate(repro_diff(), ["dndn"], canonical_placeholders=False)
        assert {"dBdi", "dBdj", "dBdidj"} <= set(signature(text))

    def test_one_gradient_argument(self):
        text = generate(repro_diff(), ["dndn"], canonical_placeholders=True)
        args = signature(text)
        assert args == ["nc", "V", "n", "B", "dBdn", "dBdndn", "f", "dfdndn"]
        assert "dBdi" not in text and "dBdj" not in text

    def test_no_duplicated_vector_temporaries(self):
        text = generate(repro_diff(), ["dndn"], canonical_placeholders=True)
        bodies = re.findall(r"^\s+t\d+\((\w)\) = (.*)$", text, re.M)
        normalised = [re.sub(rf"\b{x}\b", "#", body) for x, body in bodies]
        assert len(normalised) == len(set(normalised))

    @pytest.mark.parametrize("lowrank_hess", [False, "sym"])
    def test_hessian_is_mirrored(self, lowrank_hess):
        text = generate(
            repro_diff(),
            ["dndn"],
            canonical_placeholders=True,
            lowrank_hess=lowrank_hess,
        )
        assert "do j = i, nc" in text
        assert "dfdndn(j, i) = res" in text

    def test_hessian_of_the_placeholder_is_symmetric(self):
        text = generate(
            repro_diff(),
            ["dndn"],
            canonical_placeholders=True,
            lowrank_hess=False,
        )
        assert "dBdndn(j, i)" not in text

    def test_standard_preset(self):
        text = generate(
            repro_diff(),
            ["dn", "dndn"],
            "standard",
            canonical_placeholders=True,
        )
        assert "dBdn" in signature(text) and "dBdi" not in text


class TestNameClash:
    def test_parameter_clashing_with_canonical_name_is_reported(self):
        flux = idx_function("B")(n)
        clash = sp.Symbol("DBDN")
        expr = sum_components(n[l], l) * sp.log(1 - flux / V) * clash
        diff = DiffPlz(expr, internal_functions=[flux], indexes=[l], name="f")
        with pytest.raises(ValueError, match="same identifier"):
            to_fortran(diff, derivatives=["dndn"])


class TestMixed:
    @pytest.mark.parametrize("chain_rule", [False, True])
    def test_mixed_derivative_name(self, chain_rule):
        text = generate(
            mixed_diff(),
            ["dT", "dn", "dndT", "dndn"],
            canonical_placeholders=True,
            chain_rule=chain_rule,
        )
        args = signature(text)
        assert {"dDdn", "dDdT", "dDdTdn", "dDdndn"} <= set(args)
        assert not [a for a in args if re.fullmatch(r"dD\w*d[ij]\w*", a)]


DRIVER = """
program drive
   use kernels
   implicit none
   integer, parameter :: nc = 4
   real(8) :: n(nc), g(nc), h(nc, nc), f0, f1, h0(nc, nc), h1(nc, nc)
   integer :: a, b
   do a = 1, nc
      n(a) = 0.3d0 + 0.2d0*a
      g(a) = 0.01d0*a + 0.02d0
      do b = 1, nc
         h(a, b) = 0.001d0*(a + b) + 0.0002d0*a*b
      end do
   end do
   call legacy(nc=nc, V=2d0, n=n, B=0.1d0, dBdi=g, dBdj=g, dBdidj=h, &
               f=f0, dfdndn=h0)
   call canon(nc=nc, V=2d0, n=n, B=0.1d0, dBdn=g, dBdndn=h, f=f1, dfdndn=h1)
   write(*, '(es25.17)') abs(f1 - f0)/abs(f0), &
      maxval(abs(h1 - h0))/maxval(abs(h0))
end program drive
"""


@pytest.mark.slow
@pytest.mark.parametrize("preset", ["standard", "fastest"])
def test_same_values_as_the_legacy_kernel(preset, tmp_path):
    from fortran_tools import (
        GFORTRAN,
        OPTIMISED_FREE_FORM_FLAGS,
        compile_and_run,
    )

    if not GFORTRAN:
        pytest.skip("gfortran not available")
    diff = repro_diff()
    sources = []
    for name, flag in (("legacy", False), ("canon", True)):
        options = CodegenOptions.from_preset(
            preset, canonical_placeholders=flag
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sources.append(
                to_fortran(
                    diff,
                    derivatives=["dndn"],
                    subroutine_name=name,
                    options=options,
                )
            )
    module = "module kernels\nimplicit none\ncontains\n"
    module += "".join(sources) + "end module kernels\n"
    path = tmp_path / "drive.f90"
    path.write_text(module + DRIVER)
    run = compile_and_run([path], tmp_path, OPTIMISED_FREE_FORM_FLAGS)
    errors = [float(x) for x in run.stdout.split()]
    assert max(errors) < 1e-14


NC = 3
COEFFS = (0.7, 1.3, 0.4)
MOLES = (0.3, 0.5, 0.9)
TEMPERATURE = 2.5
VOLUME = 6.0
IN_ARGS = {
    "B": "V={vol}d0, B={b}d0, dBdn=g, dBdndn=h",
    "D": "T={temp}d0, D={b}d0, dDdT={dt}d0, dDdn=g, dDdTdn=gt, dDdndn=h",
}
OUT_ARGS = {
    "B": "f=f, dfdn=dfdn, dfdndn=dfdndn",
    "D": "f=f, dfdT=dfdT, dfdn=dfdn, dfdndn=dfdndn, dfdndT=dfdndT",
}
OUT_PRINT = {
    "B": "f, dfdn, dfdndn",
    "D": "f, dfdT, dfdn, dfdndn, dfdndT",
}

NUMERIC_DRIVER = """
program drive
   use kernels
   implicit none
   integer, parameter :: nc = {nc}
   real(8) :: n(nc), g(nc), gt(nc), h(nc, nc)
   real(8) :: f, dfdT, dfdn(nc), dfdndn(nc, nc), dfdndT(nc)
   n = [{moles}]
   g = [{grad}]
   gt = [{grad_t}]
   h = reshape([{hess}], [nc, nc])
   call kern(nc=nc, n=n, {inputs}, &
             {outputs})
   write(*, "(*(es25.17))") {printed}
end program drive
"""


def _literal(values):
    return ", ".join(f"{float(x)!r}d0" for x in values)


def _numeric_case(name):
    """Return the diff, derivatives, driver fields and SymPy values."""
    ns = sp.symbols("n0:3")
    temp = sp.Symbol("T")
    coeff = [sp.Float(c) for c in COEFFS]
    quad = sum(c * x**2 for c, x in zip(coeff, ns))
    point = {**dict(zip(ns, MOLES)), temp: TEMPERATURE}
    if name == "B":
        placeholder, expr = quad, quad
        f_expr = sum(ns) * sp.log(1 - expr / VOLUME)
        diff, ders = repro_diff(), ["dn", "dndn"]
        order = [()]
    else:
        placeholder = quad * temp**2
        f_expr = sum(ns) * sp.log(placeholder) * temp
        diff = mixed_diff()
        ders = ["dT", "dn", "dndn", "dndT"]
        order = [(temp,)]
    at = lambda e: float(e.subs(point))  # noqa: E731
    fields = {
        "b": at(placeholder),
        "dt": at(sp.diff(placeholder, temp)),
        "grad": _literal(at(sp.diff(placeholder, x)) for x in ns),
        "grad_t": _literal(at(sp.diff(placeholder, x, temp)) for x in ns),
        "hess": _literal(
            at(sp.diff(placeholder, x, y)) for y in ns for x in ns
        ),
    }
    values = [at(f_expr)]
    if order[0]:
        values.append(at(sp.diff(f_expr, temp)))
    values += [at(sp.diff(f_expr, x)) for x in ns]
    values += [at(sp.diff(f_expr, y, x)) for y in ns for x in ns]
    if order[0]:
        values += [at(sp.diff(f_expr, x, temp)) for x in ns]
    return diff, ders, fields, values


@pytest.mark.slow
@pytest.mark.parametrize("name", ["B", "D"])
@pytest.mark.parametrize("preset", ["minimal", "standard", "fastest"])
def test_kernel_matches_sympy(name, preset, tmp_path):
    """Gradients, Hessians and mixed ``dT dn`` derivatives are right."""
    from fortran_tools import (
        GFORTRAN,
        OPTIMISED_FREE_FORM_FLAGS,
        compile_and_run,
    )

    if not GFORTRAN:
        pytest.skip("gfortran not available")
    diff, derivatives, fields, expected = _numeric_case(name)
    options = CodegenOptions.from_preset(preset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        source = to_fortran(
            diff,
            derivatives=derivatives,
            subroutine_name="kern",
            options=options,
        )
    driver = NUMERIC_DRIVER.format(
        nc=NC,
        moles=_literal(MOLES),
        inputs=IN_ARGS[name].format(temp=TEMPERATURE, vol=VOLUME, **fields),
        outputs=OUT_ARGS[name],
        printed=OUT_PRINT[name],
        **fields,
    )
    module = f"module kernels\nimplicit none\ncontains\n{source}"
    path = tmp_path / "drive.f90"
    path.write_text(module + "end module kernels\n" + driver)
    run = compile_and_run([path], tmp_path, OPTIMISED_FREE_FORM_FLAGS)
    values = [float(x) for x in run.stdout.split()]
    assert values == pytest.approx(expected, rel=1e-10)
