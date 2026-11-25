"""Workload mode: fixed-output kernels behind one dispatcher (issue #11).

The numeric tests compile the generated module with gfortran and compare the
dispatcher, called with every subset of the outputs, with the optional
(fallback) kernel and with the NRTL formula written in NumPy.
"""

import pathlib
import warnings

from fortran_tools import GFORTRAN, compile_and_run

import numpy as np

import pytest

from thermodiff.codegen.workload import (
    _split_top_level,
    parse_kernel,
    specialise_kernel,
    workload_module,
    wrap_module,
)
from thermodiff.model import compile_model
from thermodiff.model.cli import main
from thermodiff.model.emit import build_diff
from thermodiff.model.errors import ModelError
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

MODEL = pathlib.Path(__file__).parent / "data/models/nrtl/nrtl.model"
WORKLOADS = ["dn,dndn", "dT"]
SIZES = (2, 3, 5)
OUTPUTS = ("f", "dfdT", "dfdn", "dfdndn")
GAS_CONSTANT = 0.0831446261815324
TEMPERATURE = 320.0
TOLERANCE = 1e-12
FLAGS = ("-O1", "-ffree-line-length-0", "-fcheck=all")

DRIVER = """
module yaeos__constants
   implicit none
   real(8), parameter :: R = 0.0831446261815324d0
end module

module kinds
   implicit none
   integer, parameter :: dp = 8
end module

{code}

program drive
   use nrtl_mod
   implicit none
   integer, parameter :: sizes(3) = [2, 3, 5]
   integer :: nc, mask, unit, idx, i, j
   real(8) :: T
   real(8), allocatable :: a(:, :), b(:, :), c(:, :), n(:)
   real(8), allocatable :: f1, dt1, dn1(:), dndn1(:, :)
   real(8), allocatable :: f2, dt2, dn2(:), dndn2(:, :)

   open(newunit=unit, file='out.txt')
   T = 320d0
   do idx = 1, size(sizes)
      nc = sizes(idx)
      allocate(a(nc, nc), b(nc, nc), c(nc, nc), n(nc))
      do i = 1, nc
         n(i) = 0.5d0 + 0.2d0*i
         do j = 1, nc
            a(i, j) = 0.1d0*(i - j) + 0.02d0*i*j
            b(i, j) = 40d0*(i + j)
            c(i, j) = 0.3d0 + 0.01d0*(i + j)
         end do
      end do
      do mask = 0, 15
         call prepare(mask)
         call nrtl_derivs(nc=nc, T=T, a=a, b=b, c=c, n=n, &
            f=f1, dfdT=dt1, dfdn=dn1, dfdndn=dndn1)
         call nrtl_derivs_fallback(nc=nc, T=T, a=a, b=b, c=c, n=n, &
            f=f2, dfdT=dt2, dfdn=dn2, dfdndn=dndn2)
         write(unit, '(a, 2(1x, i0))') 'D', nc, mask
         call dump(f1, dt1, dn1, dndn1)
         write(unit, '(a, 2(1x, i0))') 'F', nc, mask
         call dump(f2, dt2, dn2, dndn2)
         if (mask == 13) then
            call nrtl_derivs_w1(nc=nc, T=T, a=a, b=b, c=c, n=n, &
               f=f2, dfdn=dn2, dfdndn=dndn2)
         else if (mask == 3) then
            call nrtl_derivs_w2(nc=nc, T=T, a=a, b=b, c=c, n=n, &
               f=f2, dfdT=dt2)
         else
            cycle
         end if
         write(unit, '(a, 2(1x, i0))') 'G', nc, mask
         call dump(f2, dt2, dn2, dndn2)
      end do
      deallocate(a, b, c, n)
   end do
   close(unit)

contains

   subroutine prepare(bits)
      integer, intent(in) :: bits
      if (allocated(f1)) deallocate(f1, f2)
      if (allocated(dt1)) deallocate(dt1, dt2)
      if (allocated(dn1)) deallocate(dn1, dn2)
      if (allocated(dndn1)) deallocate(dndn1, dndn2)
      if (btest(bits, 0)) allocate(f1, f2)
      if (btest(bits, 1)) allocate(dt1, dt2)
      if (btest(bits, 2)) allocate(dn1(nc), dn2(nc))
      if (btest(bits, 3)) allocate(dndn1(nc, nc), dndn2(nc, nc))
   end subroutine prepare

   subroutine dump(f, dt, dn, dndn)
      real(8), allocatable, intent(in) :: f, dt, dn(:), dndn(:, :)
      if (allocated(f)) write(unit, '(es30.17e3)') f
      if (allocated(dt)) write(unit, '(es30.17e3)') dt
      if (allocated(dn)) write(unit, '(es30.17e3)') dn
      if (allocated(dndn)) write(unit, '(es30.17e3)') dndn
   end subroutine dump
end program
"""


def generate(**kwargs):
    """Compile the NRTL model in workload mode."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compile_model(
            MODEL.read_text(),
            workloads=WORKLOADS,
            derivatives="dT,dn,dndn",
            **kwargs,
        )


def run_driver(code, tmp_path):
    """Compile ``code`` with the driver and return its values.

    Returns
    -------
    dict
        ``(source, nc, mask)`` to the numbers written, with ``source``
        ``"D"`` (dispatcher), ``"F"`` (fallback) or ``"G"`` (the generic
        fixed-output kernel, only for the masks of the workloads).
    """
    source = tmp_path / "drive.f90"
    source.write_text(DRIVER.format(code=code))
    compile_and_run([source], tmp_path, FLAGS)
    values = {}
    key = None
    for line in (tmp_path / "out.txt").read_text().splitlines():
        words = line.split()
        if words[0] in ("D", "F", "G"):
            key = (words[0], int(words[1]), int(words[2]))
            values[key] = []
        else:
            values[key].append(float(line))
    return {key: np.array(value) for key, value in values.items()}


def nrtl_reference(nc):
    """Return the NRTL excess Gibbs energy of the driver's inputs."""
    ids = np.arange(1, nc + 1)
    n = 0.5 + 0.2 * ids
    i, j = np.meshgrid(ids, ids, indexing="ij")
    a = 0.1 * (i - j) + 0.02 * i * j
    b = 40.0 * (i + j)
    c = 0.3 + 0.01 * (i + j)
    tau = a + b / TEMPERATURE
    gij = np.exp(-c * tau)
    numerator = (n[:, None] * tau * gij).sum(axis=0)
    denominator = (n[:, None] * gij).sum(axis=0)
    return GAS_CONSTANT * TEMPERATURE * (n * numerator / denominator).sum()


class TestHelpers:
    def test_split_top_level_keeps_dimensions_together(self):
        text = "a(:, :), n(nc), t"
        assert _split_top_level(text) == ["a(:, :)", "n(nc)", "t"]

    def test_parse_kernel_reads_dummies(self):
        code = generate(preset="minimal")
        iface = parse_kernel(code)
        assert iface.name == "nrtl_derivs"
        assert iface.outputs == list(OUTPUTS)
        assert iface.dummies["a"].rank == 2
        assert iface.dummies["dfdn"].is_optional
        assert not iface.has_extra_dimensions

    def test_extra_dimensions_are_forwarded_with_their_extents(self):
        source = "\n".join(
            [
                "subroutine kern(nc, ng, v, n, f, dfdn)",
                "   implicit none",
                "   integer, intent(in) :: nc",
                "   integer, intent(in) :: ng",
                "   real(8), contiguous, intent(in) :: v(:, :)",
                "   real(8), contiguous, intent(in) :: n(:)",
                "   real(8), intent(out) :: f",
                "   real(8), contiguous, intent(out) :: dfdn(:)",
                "   f = 0d0",
                "end subroutine kern",
            ]
        )
        iface = parse_kernel(source)
        assert iface.has_extra_dimensions
        unknown = specialise_kernel(iface, name="k", extents={})
        assert "real(8), intent(in) :: v(:, :)" in unknown
        assert "real(8), intent(out) :: dfdn(nc)" in unknown
        known = specialise_kernel(
            iface, name="k", extents={"V": ("nc", "ng"), "n": ("nc",)}
        )
        assert "real(8), intent(in) :: v(nc, ng)" in known
        assert "real(8), intent(in) :: n(nc)" in known

    def test_wrap_module_indents_the_code(self):
        text = wrap_module("m", ["k, only: dp"], "subroutine s\n\nend")
        assert text.splitlines() == [
            "module m",
            "   use k, only: dp",
            "   implicit none",
            "contains",
            "",
            "   subroutine s",
            "",
            "   end",
            "end module m",
        ]


@pytest.fixture(scope="module")
def code():
    """Module of the ``fastest`` preset, which has contiguous inputs."""
    return generate(preset="fastest")


class TestStructure:
    """Text facts of the module; the numbers are checked below."""

    def dispatcher(self, code):
        start = code.index("subroutine nrtl_derivs(")
        return code[start : code.index("end subroutine nrtl_derivs\n")]

    def test_dispatcher_has_the_optional_signature(self, code):
        dispatcher = self.dispatcher(code)
        assert "subroutine nrtl_derivs(nc, T, a, b, c, n, f, dfdT, " in code
        assert "optional, intent(out) :: dfdn(:)" in dispatcher
        assert "contiguous, optional" not in dispatcher
        assert "real(8), contiguous, intent(in) :: a(:, :)" in dispatcher

    def test_one_branch_per_workload_and_a_default(self, code):
        dispatcher = self.dispatcher(code)
        assert "case (13)  ! f, dfdn, dfdndn" in dispatcher
        assert "case (3)  ! f, dfdT" in dispatcher
        assert "case default" in dispatcher
        assert "call nrtl_derivs_fallback(" in dispatcher

    def test_kernels_are_explicit_shape_and_not_inlined(self, code):
        start = code.index("subroutine nrtl_derivs_w1(")
        kernel = code[start : code.index("end subroutine nrtl_derivs_w1")]
        assert "real(8), intent(in) :: a(nc, nc)" in kernel
        assert "real(8), intent(out) :: dfdndn(nc, nc)" in kernel
        assert "contiguous" not in kernel
        assert "!GCC$ ATTRIBUTES NOINLINE :: nrtl_derivs_w1" in kernel
        assert "optional" not in kernel

    def test_fallback_keeps_the_input_contiguity(self, code):
        start = code.index("subroutine nrtl_derivs_fallback(")
        fallback = code[start : code.index("real(8) :: res", start)]
        assert "contiguous, intent(in) :: n(:)" in fallback
        assert "contiguous, optional" not in fallback

    def test_equal_output_sets_are_rejected(self):
        with pytest.raises(ModelError, match="same output set"):
            compile_model(
                MODEL.read_text(),
                workloads=["dT", "dT"],
                preset="minimal",
            )

    def test_workload_outside_the_fallback_is_rejected(self):
        ir = build_ir(read_sections(MODEL.read_text(), None), None)
        diff, _ = build_diff(ir, "dT,dn")
        with pytest.raises(ValueError, match="fallback"):
            workload_module(
                diff,
                ["dn"],
                derivatives="dT",
                subroutine_name="nrtl_derivs",
                module_symbols=ir.imported,
                preset="minimal",
            )

    def test_optional_derivatives_is_not_accepted(self):
        with pytest.raises(ModelError, match="workload mode"):
            compile_model(
                MODEL.read_text(),
                workloads=["dT"],
                optional_derivatives=True,
            )

    def test_cli_workload_flag(self, tmp_path, capsys):
        target = tmp_path / "nrtl.f90"
        status = main(
            [
                str(MODEL),
                "--preset",
                "minimal",
                "--workload",
                "dT",
                "--workload",
                "dn,dndn",
                "--derivatives",
                "dT,dn,dndn",
                "-o",
                str(target),
            ]
        )
        assert status == 0
        assert "case (3)  ! f, dfdT" in target.read_text()


@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
class TestNumeric:
    CONFIGS = {
        "fastest": {"preset": "fastest"},
        "standard": {"preset": "standard"},
        "work_array": {"preset": "fastest", "work_array": True},
        "real_kind": {
            "preset": "standard",
            "real_kind": "dp",
            "kind_module": "kinds",
        },
    }

    @pytest.mark.parametrize("config", sorted(CONFIGS))
    def test_dispatcher_equals_the_optional_kernel(self, config, tmp_path):
        values = run_driver(generate(**self.CONFIGS[config]), tmp_path)
        for nc in SIZES:
            for mask in range(1 << len(OUTPUTS)):
                got = values[("D", nc, mask)]
                want = values[("F", nc, mask)]
                assert got.shape == want.shape
                scale = max(np.linalg.norm(want), 1.0)
                assert np.linalg.norm(got - want) <= TOLERANCE * scale

    def test_workload_sets_run_the_fixed_output_kernel(self, tmp_path):
        values = run_driver(generate(preset="fastest"), tmp_path)
        for nc in SIZES:
            for mask in (0b1101, 0b0011):
                got = values[("D", nc, mask)]
                assert np.array_equal(got, values[("G", nc, mask)])

    def test_value_matches_the_formula(self, tmp_path):
        values = run_driver(generate(preset="fastest"), tmp_path)
        for nc in SIZES:
            got = values[("D", nc, 0b1111)][0]
            assert got == pytest.approx(nrtl_reference(nc), rel=1e-12)

    def test_absent_arrays_do_not_crash(self, tmp_path):
        # Unallocated allocatables are absent optional actuals.  With
        # ``contiguous`` on the dispatcher's optional dummies gfortran may
        # try to pack them (issue 11, pitfall 1); the fallback of
        # ``work_array`` kernels, whose internal body has its own optional
        # dummies, crashed that way.  The driver runs every mask with
        # -fcheck=all, so reaching the end of run_driver is the test.
        values = run_driver(generate(preset="fastest"), tmp_path)
        assert values[("D", 3, 0b0100)].shape == (3,)
        assert values[("D", 3, 0)].shape == (0,)
