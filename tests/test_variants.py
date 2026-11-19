"""Tests for the V1/V2/V3 generation variants and the ``share`` flag.

V1 "none" = ``array_cse=False, share=False`` (D-007), V2 "partial" =
``array_cse=False``, V3 "full" = ``array_cse=True``.  The three variants
must emit identical interfaces and equivalent bodies.
"""

import pytest

from thermodiff import DiffPlz, T, V, k, n, sum_components, to_fortran
from thermodiff.model import compile_model
from thermodiff.model.cli import main as cli_main

from fortran_tools import GFORTRAN, compile_and_run

V1 = {"array_cse": False, "share": False}
V2 = {"array_cse": False, "share": True}
V3 = {"array_cse": True, "share": True}
VARIANTS = {"v1": V1, "v2": V2, "v3": V3}


def toy_diff():
    """Return a DiffPlz for f = T * sum(n_k**2) + V * sum(n_k)."""
    expr = T * sum_components(n[k] ** 2, k) + V * sum_components(n[k], k)
    return DiffPlz(expr, internal_functions=None, indexes=[k], name="f")


SMALL_MODEL = """
model small
variables
    n nc
    V
    T
parameters
    b nc
expressions
    bmix = sum_k n_k*b_k
function
    T*bmix^2 + V*bmix
"""


def _interface(code):
    """Return (subroutine statement, input decls, output decls)."""
    lines = code.splitlines()
    signature = []
    for line in lines:
        signature.append(line.strip())
        if line.rstrip().endswith(")"):
            break
    inputs = [line.strip() for line in lines if "intent(in)" in line]
    outputs = [line.strip() for line in lines if "intent(out)" in line]
    return " ".join(signature), inputs, outputs


class TestShareFlag:
    """The share flag toggles only the sharing passes."""

    def test_diffplz_forwards_share_kwarg(self):
        diff = toy_diff()
        assert diff.to_fortran(share=False, array_cse=False) == to_fortran(
            diff, share=False, array_cse=False
        )
        assert diff.to_fortran(share=True, array_cse=True) == to_fortran(diff)

    def test_share_defaults_to_true(self):
        diff = toy_diff()
        assert to_fortran(diff) == to_fortran(diff, share=True)
        assert to_fortran(diff) == to_fortran(diff, share=True, array_cse=True)

    def test_v1_repeats_shared_sums(self):
        diff = toy_diff()
        v1 = to_fortran(diff, **V1)
        v2 = to_fortran(diff, **V2)
        v3 = to_fortran(diff, **V3)
        assert v1.count("do k = 1, nc") > v2.count("do k = 1, nc")
        assert v2.count("do k = 1, nc") == 2
        assert v3.count("do k = 1, nc") == 2

    def test_v1_loop_indices_are_declared(self):
        # Regression: with share=False the sums live inside the outputs
        # and their loop indices must still be declared integers.
        code = to_fortran(toy_diff(), derivatives="none,dT,dV,dn", **V1)
        assert "integer :: i, k" in code

    def test_emit_module_forwards_share(self):
        default = compile_model(SMALL_MODEL, filename="small.model")
        unshared = compile_model(SMALL_MODEL, filename="small.model", **V1)
        assert unshared.count("do k = 1, nc") > default.count("do k = 1, nc")
        assert _interface(unshared)[0] == _interface(default)[0]

    def test_cli_no_share_flag(self, tmp_path, capsys):
        model_path = tmp_path / "small.model"
        model_path.write_text(SMALL_MODEL)
        assert cli_main([str(model_path), "--no-share", "--no-array-cse"]) == 0
        emitted = capsys.readouterr().out
        expected = compile_model(SMALL_MODEL, filename="small.model", **V1)
        assert emitted == expected


class TestStatsHook:
    """The ``stats`` dict reports operation counts and temporaries."""

    def test_stats_keys(self):
        stats = {}
        to_fortran(toy_diff(), stats=stats)
        assert set(stats) == {
            "operation_count",
            "operation_count_tool",
            "n_temporaries",
        }
        assert stats["operation_count_tool"] == "sympy.count_ops(visual=False)"
        assert isinstance(stats["operation_count"], int)

    def test_stats_reflect_sharing(self):
        counts = {}
        for name, settings in VARIANTS.items():
            stats = {}
            to_fortran(toy_diff(), stats=stats, **settings)
            counts[name] = stats
        assert counts["v1"]["n_temporaries"] == 0
        assert counts["v2"]["n_temporaries"] >= 1
        assert counts["v3"]["n_temporaries"] >= 1
        assert (
            counts["v1"]["operation_count"] > counts["v2"]["operation_count"]
        )

    def test_stats_none_by_default(self):
        assert to_fortran(toy_diff()) == to_fortran(toy_diff(), stats=None)


class TestVariantInterfaces:
    """Signatures do not depend on the variant or on the optional flag."""

    @pytest.mark.parametrize(
        "workload",
        ["none", "dn", "dn,dndV,dndT", "dT,dV,dn,dT2,dV2,dndn,dTdV,dndT,dndV"],
    )
    def test_headers_identical_across_variants(self, workload):
        diff = toy_diff()
        headers = {}
        for name, settings in VARIANTS.items():
            code = to_fortran(
                diff,
                derivatives=workload,
                optional_derivatives=True,
                **settings,
            )
            headers[name] = _interface(code)
        assert headers["v1"] == headers["v2"]
        assert headers["v2"] == headers["v3"]

    @pytest.mark.parametrize(
        "workload",
        ["dn", "dn,dndV,dndT", "dT,dV,dn,dT2,dV2,dndn,dTdV,dndT,dndV"],
    )
    def test_optional_order_matches_plain_order(self, workload):
        diff = toy_diff()
        plain = _interface(to_fortran(diff, derivatives=workload))
        optional = _interface(
            to_fortran(diff, derivatives=workload, optional_derivatives=True)
        )
        assert plain[0] == optional[0]
        assert plain[1] == optional[1]
        assert optional[2]
        for line in optional[2]:
            assert line.startswith("real(8), optional, intent(out)")
        for line in plain[2]:
            assert line.startswith("real(8), intent(out)")

    def test_optional_guards_outputs_and_temporaries(self):
        diff = toy_diff()
        code = to_fortran(
            diff, derivatives="dT,dV,dn", optional_derivatives=True, **V2
        )
        assert "if (present(f)) then" in code
        assert "if (present(dfdn)) then" in code
        # Each shared temporary is guarded by the union of its consumers.
        assert "if (present(f) .or. present(dfdT)) then" in code
        assert "if (present(f) .or. present(dfdV)) then" in code

    def test_determinism(self):
        diff = toy_diff()
        for settings in VARIANTS.values():
            first = to_fortran(diff, optional_derivatives=True, **settings)
            second = to_fortran(diff, optional_derivatives=True, **settings)
            assert first == second


@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
class TestCompileVariants:
    """Compile and run the variants; N is a runtime argument."""

    def _compile_and_run(self, tmp_path, code, driver):
        module = "module toy_mod\ncontains\n" + code
        module += "\nend module toy_mod\n"
        source = tmp_path / "main.f90"
        source.write_text(module + driver)
        run = compile_and_run([source], tmp_path, ("-O2",))
        return [float(value) for value in run.stdout.split()]

    def test_interface_is_independent_of_nc(self, tmp_path):
        # One compilation, two calls with nc = 2 and nc = 4.
        code = to_fortran(toy_diff(), derivatives="none,dT,dV,dn")
        driver = (
            "program main\n"
            "  use toy_mod\n"
            "  implicit none\n"
            "  real(8) :: n2(2), n4(4)\n"
            "  real(8) :: V, T, f, dfdT, dfdV\n"
            "  real(8) :: dfdn2(2), dfdn4(4)\n"
            "  n2 = [1.0d0, 2.0d0]\n"
            "  n4 = [0.5d0, 1.0d0, 1.5d0, 2.0d0]\n"
            "  V = 3.0d0\n"
            "  T = 1.5d0\n"
            "  call f_derivs(2, V, T, n2, f, dfdT, dfdV, dfdn2)\n"
            "  write(*,*) f, dfdT, dfdV\n"
            "  write(*,*) dfdn2\n"
            "  call f_derivs(4, V, T, n4, f, dfdT, dfdV, dfdn4)\n"
            "  write(*,*) f, dfdT, dfdV\n"
            "  write(*,*) dfdn4\n"
            "end program main\n"
        )
        values = self._compile_and_run(tmp_path, code, driver)
        expected = []
        for nvec in ([1.0, 2.0], [0.5, 1.0, 1.5, 2.0]):
            s1 = sum(nvec)
            s2 = sum(x * x for x in nvec)
            expected.extend([1.5 * s2 + 3.0 * s1, s2, s1])
            expected.extend(2 * 1.5 * x + 3.0 for x in nvec)
        assert values == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("settings", [V1, V2, V3])
    def test_optional_subset_agrees(self, tmp_path, settings):
        code = to_fortran(
            toy_diff(),
            derivatives="none,dT,dV,dn",
            optional_derivatives=True,
            **settings,
        )
        driver = (
            "program main\n"
            "  use toy_mod\n"
            "  implicit none\n"
            "  real(8) :: n(3), V, T\n"
            "  real(8) :: f, dfdT, dfdV, dfdn(3), only_n(3)\n"
            "  n = [1.0d0, 2.0d0, 3.0d0]\n"
            "  V = 5.0d0\n"
            "  T = 2.0d0\n"
            "  call f_derivs(3, V, T, n, f=f)\n"
            "  write(*,*) f\n"
            "  call f_derivs(3, V, T, n, f, dfdT, dfdV, dfdn)\n"
            "  write(*,*) f, dfdT, dfdV\n"
            "  write(*,*) dfdn\n"
            "  call f_derivs(3, V, T, n, dfdn=only_n)\n"
            "  write(*,*) only_n\n"
            "end program main\n"
        )
        values = self._compile_and_run(tmp_path, code, driver)
        assert values[0] == pytest.approx(58.0)  # f alone
        assert values[1] == pytest.approx(58.0)  # f with everything
        assert values[2] == pytest.approx(14.0)  # dfdT
        assert values[3] == pytest.approx(6.0)  # dfdV
        assert values[4:7] == pytest.approx([9.0, 13.0, 17.0])  # dfdn
        assert values[7:10] == pytest.approx([9.0, 13.0, 17.0])  # dfdn alone

    def test_variants_agree_numerically(self, tmp_path):
        results = {}
        for name, settings in VARIANTS.items():
            code = to_fortran(
                toy_diff(),
                derivatives="dT,dV,dn,dndn,dndT,dndV",
                optional_derivatives=True,
                **settings,
            )
            driver = (
                "program main\n"
                "  use toy_mod\n"
                "  implicit none\n"
                "  real(8) :: n(3), V, T\n"
                "  real(8) :: f, dfdT, dfdV, dfdndV\n"
                "  real(8) :: dfdn(3), dfdndT(3), dfdndn(3, 3)\n"
                "  n = [1.0d0, 2.0d0, 3.0d0]\n"
                "  V = 5.0d0\n"
                "  T = 2.0d0\n"
                "  call f_derivs(3, V, T, n, f, dfdT, dfdV, dfdn, dfdndn, "
                "dfdndT, dfdndV)\n"
                "  write(*,*) f, dfdT, dfdV\n"
                "  write(*,*) dfdn\n"
                "  write(*,*) dfdndT\n"
                "  write(*,*) dfdndV\n"
                "  write(*,*) dfdndn(1, 1), dfdndn(1, 2)\n"
                "end program main\n"
            )
            results[name] = self._compile_and_run(tmp_path, code, driver)
        assert results["v1"] == pytest.approx(results["v2"], rel=1e-12)
        assert results["v2"] == pytest.approx(results["v3"], rel=1e-12)
        # sanity: f = T*sum(n**2) + V*sum(n)
        assert results["v1"][0] == pytest.approx(58.0)
