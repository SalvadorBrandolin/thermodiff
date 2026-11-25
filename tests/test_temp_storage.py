"""Storage of the array temporaries: ``work_array`` and ``max_nc`` (#10).

Without ``-fstack-arrays`` gfortran allocates every automatic array on the
heap.  ``work_array`` puts every temporary in one work array (one
allocation per call); ``max_nc`` gives them compile-time extents (stack,
no allocation) behind an ``nc > max_nc`` guard.  Both change only the
declarations, so the results must be bit-identical to the default kernel.
"""

import os
import pathlib
import re
import tempfile
import warnings

from fortran_tools import DEBUG_FREE_FORM_FLAGS, GFORTRAN, compile_and_run

import numpy as np

import pytest

import sympy as sp

from thermodiff import DiffPlz, T, idx_function, k, l, n, sum_components
from thermodiff.codegen.options import CodegenOptions
from thermodiff.codegen.storage import (
    GFORTRAN_MAX_STACK_VAR_SIZE,
    array_declarations,
    work_array_kernel,
)
from thermodiff.fortran import to_fortran

_MODELS = os.path.join(os.path.dirname(__file__), "data", "models")
DERIVS = ["dn", "dndT", "dndn"]


def nrtl_diff():
    """NRTL-like G^E with an indexed placeholder ``tau(l, k, T)``."""
    alpha = sp.IndexedBase("alpha")
    tau = idx_function("tau")(l, k, T)
    e_lk = sp.exp(-alpha[l, k] * tau)
    num = sum_components(n[l] * e_lk * tau, l)
    den = sum_components(n[l] * e_lk, l)
    ge = sp.Symbol("R") * T * sum_components(n[k] * num / den, k)
    return DiffPlz(ge, internal_functions=[tau], indexes=[k, l], name="ge")


def generate(**extra):
    """Generate the fastest fixed-output kernel with ``extra`` options."""
    options = CodegenOptions.from_preset(
        "fastest", optional_derivatives=False, chain_rule=False, **extra
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return to_fortran(
            nrtl_diff(),
            derivatives=DERIVS,
            subroutine_name="kern",
            module_symbols=["R"],
            options=options,
        )


class TestOptions:
    def test_defaults_are_off(self):
        options = CodegenOptions()
        assert options.work_array is False
        assert options.max_nc is None
        for preset in ("minimal", "standard", "fastest"):
            options = CodegenOptions.from_preset(preset)
            assert not options.work_array and options.max_nc is None

    @pytest.mark.parametrize("value", [None, False])
    def test_max_nc_off_values(self, value):
        assert CodegenOptions(max_nc=value).max_nc is None

    def test_max_nc_is_coerced_to_int(self):
        assert CodegenOptions(max_nc=16.0).max_nc == 16
        assert CodegenOptions(max_nc=16).to_legacy()["max_nc"] == 16

    @pytest.mark.parametrize("value", [0, -3, 2.5, True])
    def test_max_nc_rejects_non_positive_integers(self, value):
        with pytest.raises(ValueError, match="max_nc"):
            CodegenOptions(max_nc=value)

    def test_max_nc_disables_work_array(self):
        options = CodegenOptions(work_array=True, max_nc=8)
        with pytest.warns(UserWarning, match="work_array"):
            normalised, names = options.validate()
        assert names == ("work_array",)
        assert normalised.max_nc == 8 and not normalised.work_array

    def test_work_array_alone_is_valid(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            normalised, names = CodegenOptions(work_array=True).validate()
        assert names == () and normalised.work_array

    def test_to_fortran_keyword(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = to_fortran(nrtl_diff(), derivatives=DERIVS, max_nc=4)
        assert "error stop" in text


class TestDeclarations:
    def test_default_kernel_has_nc_sized_temporaries(self):
        text = generate()
        assert re.search(r"real\(8\) :: t\d+\(nc, nc\)", text)
        assert "work_" not in text and "error stop" not in text

    def test_max_nc_gives_literal_extents_and_a_guard(self):
        text = generate(max_nc=12)
        assert "(nc)" not in text and "(nc, nc)" not in text
        assert re.search(r"real\(8\) :: t\d+\(12, 12\)", text)
        assert "if (nc > 12) error stop 'kern: nc > max_nc = 12'" in text

    def test_max_nc_with_array_pad_pads_the_leading_extent(self):
        lines = array_declarations({"t1": 1, "t2": 2}, 1, 8)
        assert lines == ["   real(8) :: t1(9)", "   real(8) :: t2(9, 8)"]

    def test_large_max_nc_warns_about_static_storage(self):
        with pytest.warns(UserWarning, match="fmax-stack-var-size"):
            array_declarations({"t1": 2}, 0, 91)
        assert 90 * 90 * 8 <= GFORTRAN_MAX_STACK_VAR_SIZE

    def test_work_array_declares_one_array_and_an_internal_body(self):
        text = generate(work_array=True)
        head, inner = text.split("contains\n")
        assert len(re.findall(r"real\(8\) :: \w+\(", head)) == 1
        assert re.search(r"real\(8\) :: work_\(\d+\*nc \+ \d+\*nc\*nc\)", head)
        assert "call kern_body(" in head and "work_(1)" in head
        assert "subroutine kern_body(" in inner
        assert inner.rstrip().endswith("end subroutine kern")

    def test_work_array_offsets_follow_the_declaration_order(self):
        lines = work_array_kernel(
            "s", ["nc"], [], [], [], {"b": 2, "a": 1, "c": 1}, 2
        )
        text = "\n".join(lines)
        assert "work_(2*nc + (nc + 2)*nc)" in text
        assert "call s_body(nc, work_(1), work_(1 + nc), " in text
        assert "work_(1 + 2*nc))" in text
        assert "subroutine s_body(nc, a, c, b)" in text

    def test_work_array_names_avoid_clashes(self):
        lines = work_array_kernel(
            "s", ["nc", "work_", "s_body"], [], [], [], {"t1": 1}, 0
        )
        text = "\n".join(lines)
        assert "real(8) :: work__(nc)" in text
        assert "subroutine s_body_(nc, work_, s_body, t1)" in text

    def test_no_array_temporaries_keep_the_default_layout(self):
        diff = DiffPlz(sp.Symbol("a") * T**2, name="f")
        for extra in ({"work_array": True}, {"max_nc": 4}):
            options = CodegenOptions(**extra)
            assert to_fortran(diff, options=options) == to_fortran(diff)


def _run_model(settings, ncs=(3, 7), opt=False, flags=DEBUG_FREE_FORM_FLAGS):
    """Compile ``nrtl.model`` with ``settings`` and run it at ``ncs``."""
    from model_harness import driver
    from thermodiff.model import compile_model

    text = open(os.path.join(_MODELS, "nrtl", "nrtl.model")).read()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = compile_model(
            text,
            derivatives="dT,dn,dT2,dndn,dndT",
            optional_derivatives=opt,
            **{"preset": "fastest", **settings},
        )
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "drive.f90").write_text(driver(code, {}, _OUTS, ncs=ncs))
        compile_and_run([tmp / "drive.f90"], tmp, flags)
        return np.loadtxt(tmp / "out.txt"), code


_OUTS = ["f", "dfdT", "dfdn", "dfdT2", "dfdndn", "dfdndT"]


@pytest.mark.slow
@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
class TestNumerics:
    @pytest.mark.parametrize("settings", [{"work_array": True}, {"max_nc": 8}])
    @pytest.mark.parametrize("opt", [False, True])
    def test_same_results_as_the_default_kernel(self, settings, opt):
        ref, ref_code = _run_model({}, opt=opt)
        val, code = _run_model(settings, opt=opt)
        assert code != ref_code
        np.testing.assert_array_equal(val, ref)

    def test_max_nc_guard_stops_the_kernel(self):
        with pytest.raises(AssertionError, match="nc > max_nc = 4"):
            _run_model({"max_nc": 4}, ncs=(3, 7))

    @pytest.mark.parametrize("settings", [{"work_array": True}, {"max_nc": 8}])
    def test_same_results_without_stack_arrays(self, settings):
        flags = (*DEBUG_FREE_FORM_FLAGS[1:], "-O2", "-fno-stack-arrays")
        ref, _ = _run_model({}, flags=flags)
        val, _ = _run_model(settings, flags=flags)
        np.testing.assert_array_equal(val, ref)
