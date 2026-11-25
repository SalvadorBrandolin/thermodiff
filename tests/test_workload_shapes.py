"""Workload dispatcher with actual arrays larger than ``nc`` (and ``ng``).

The fixed-output kernels declare explicit-shape dummies (``v(nc, ng)``);
an actual array whose leading extent differs would be re-strided by
sequence association.  The dispatcher must route those calls to the
assumed-shape fallback, so its results never depend on the declared extents
of the caller's arrays.
"""

import warnings

from fortran_tools import GFORTRAN, compile_and_run

import numpy as np

import pytest

from test_extra_dimensions import GROUP_MIXED

from thermodiff.model import compile_model

pytestmark = pytest.mark.skipif(
    GFORTRAN is None, reason="gfortran is not available"
)

FLAGS = ("-O1", "-ffree-line-length-0", "-fcheck=all")
NC = 2
NG = 3
EXTRA = 2
TOLERANCE = 1e-13

DRIVER = """
{code}

program drive
   use grp3_mod
   implicit none
   integer, parameter :: nc = {nc}, ng = {ng}, lc = {lc}, lg = {lg}
   integer, parameter :: mc = {mc}
   real(8) :: n(nc), q(ng), r(nc), v(nc, ng), t
   real(8) :: nb(lc), qb(lg), rb(lc), vb(lc, lg)
   real(8) :: f(3), dt(3), dn(mc, 3), dndn(mc, mc, 3)
   integer :: i, j, k, unit

   t = 310d0
   nb = -1d9; qb = -1d9; rb = -1d9; vb = -1d9
   do i = 1, nc
      n(i) = 0.5d0 + 0.2d0*i
      r(i) = 1.0d0 + 0.3d0*i
      do j = 1, ng
         v(i, j) = 0.4d0 + 0.1d0*i*j
      end do
   end do
   do j = 1, ng
      q(j) = 0.7d0 + 0.15d0*j
   end do
   nb(1:nc) = n; rb(1:nc) = r; qb(1:ng) = q; vb(1:nc, 1:ng) = v
   dn = -1d9; dndn = -1d9

   ! 1: exact arrays through the dispatcher
   call grp3_derivs(nc=nc, ng=ng, T=t, n=n, q=q, r=r, v=v, f=f(1), &
      dfdT=dt(1), dfdn=dn(1:nc, 1), dfdndn=dndn(1:nc, 1:nc, 1))
   ! 2: larger arrays through the dispatcher
   call grp3_derivs(nc=nc, ng=ng, T=t, n=nb, q=qb, r=rb, v=vb, f=f(2), &
      dfdT=dt(2), dfdn=dn(:, 2), dfdndn=dndn(:, :, 2))
   ! 3: larger arrays through the fallback
   call grp3_derivs_fallback(nc=nc, ng=ng, T=t, n=nb, q=qb, r=rb, v=vb, &
      f=f(3), dfdT=dt(3), dfdn=dn(:, 3), dfdndn=dndn(:, :, 3))

   open(newunit=unit, file='out.txt')
   do i = 1, 3
      write(unit, '(es30.17e3)') f(i), dt(i), dn(1:nc, i), &
         ((dndn(j, k, i), j = 1, nc), k = 1, nc)
   end do
   close(unit)
end program
"""


def _generate(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compile_model(
            GROUP_MIXED,
            workloads=["dn,dndn,dT"],
            derivatives="dn,dndn,dT",
            **kwargs,
        )


def _run(code, tmp_path, extra_in, extra_out):
    source = tmp_path / "drive.f90"
    text = DRIVER.format(
        code=code,
        nc=NC,
        ng=NG,
        lc=NC + extra_in,
        lg=NG + extra_in,
        mc=NC + extra_out,
    )
    source.write_text(text)
    compile_and_run([source], tmp_path, FLAGS)
    values = np.loadtxt(tmp_path / "out.txt").reshape(3, -1)
    return values


@pytest.mark.parametrize("preset", ["minimal", "fastest"])
@pytest.mark.parametrize("fixed_nc", [None, (NC,)])
@pytest.mark.parametrize(
    "extra_in, extra_out", [(EXTRA, 0), (0, EXTRA), (EXTRA, EXTRA)]
)
def test_oversized_arrays_equal_the_fallback(
    preset, fixed_nc, extra_in, extra_out, tmp_path
):
    code = _generate(preset=preset, fixed_nc=fixed_nc)
    values = _run(code, tmp_path, extra_in, extra_out)
    exact, dispatched, fallback = values
    scale = np.linalg.norm(fallback)
    assert np.linalg.norm(dispatched - fallback) <= TOLERANCE * scale
    assert np.linalg.norm(exact - fallback) <= TOLERANCE * scale
