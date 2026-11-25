"""Tests of thermodiff.yaeos_model (yaeos ArModel/GeModel wrappers of the
kernels).

Pure Python: the emitted Fortran source is checked textually for a small case
(type name, parameter components and ranks, one ``case`` per workload mask with
the right kernel and outputs, and the default branch).
"""

import re

from fortran_tools import GFORTRAN, compile_and_run
import pytest

from thermodiff.yaeos_model import kernel_interface, yaeos_model_module

KERNEL = """subroutine toy_k1_w1(nc, V, T, a, kij, n, f, dfdV)
   implicit none
   integer, intent(in) :: nc
   real(8), intent(in) :: V
   real(8), intent(in) :: T
   real(8), contiguous, intent(in) :: a(:)
   real(8), intent(in) :: kij(:, :)
   real(8), intent(in) :: n(:)
   real(8), intent(out) :: f
   real(8), intent(out) :: dfdV
   f = 0; dfdV = 0
end subroutine toy_k1_w1
"""

AR_ALL = [
    "f",
    "dfdV",
    "dfdT",
    "dfdn",
    "dfdV2",
    "dfdT2",
    "dfdTdV",
    "dfdndn",
    "dfdndV",
    "dfdndT",
]


def _args(outputs, params=(("a", 1), ("kij", 2))):
    return (
        [("nc", 0), ("V", 0), ("T", 0)]
        + list(params)
        + [("n", 1)]
        + [
            (
                o,
                (
                    0
                    if o in ("f", "dfdV", "dfdT", "dfdV2", "dfdT2", "dfdTdV")
                    else 1
                ),
            )
            for o in outputs
        ]
    )


def _ar_workloads():
    sets = {
        "W0": ["f"],
        "W1": ["f", "dfdV"],
        "W3": ["f", "dfdn"],
        "W6": AR_ALL,
    }
    return {
        w: dict(routine=f"toy_k1_{w.lower()}", args=_args(o), outputs=o)
        for w, o in sets.items()
    }


def _call_keywords(flat_source, case_label):
    """Return the ``keyword=value`` arguments of the call under a case.

    Parameters
    ----------
    flat_source : str
        Wrapper source with continuation markers removed.
    case_label : str
        Selector of the ``case (...)`` block, for example ``"3"``.

    Returns
    -------
    dict of str to str
        Keyword argument names mapped to their value expressions, ignoring
        spacing and argument order.
    """
    block = re.search(
        rf"case \({case_label}\).*?\n\s*call \w+\((.*)\)\s*\n",
        flat_source,
    ).group(1)
    arguments, depth, current = [], 0, ""
    for char in block:
        depth += (char == "(") - (char == ")")
        if char == "," and depth == 0:
            arguments.append(current)
            current = ""
        else:
            current += char
    arguments.append(current)
    pairs = (argument.split("=", 1) for argument in arguments)
    return {name.strip(): value.strip() for name, value in pairs}


def test_kernel_interface_ranks():
    routine, args = kernel_interface(KERNEL)
    assert routine == "toy_k1_w1"
    assert args == [
        ("nc", 0),
        ("V", 0),
        ("T", 0),
        ("a", 1),
        ("kij", 2),
        ("n", 1),
        ("f", 0),
        ("dfdV", 0),
    ]


def test_ar_wrapper_source():
    src = yaeos_model_module(
        "toy",
        "k1",
        "ar",
        _ar_workloads(),
        "td_toy_k1",
        v0_expr="sum(n*self%a)/sum(n)",
    )
    assert src.startswith("module td_toy_k1_model")
    assert "type, extends(ArModel) :: TD_TOY_K1" in src
    assert "real(pr), allocatable :: a(:)" in src
    assert "real(pr), allocatable :: kij(:, :)" in src
    assert "procedure :: residual_helmholtz => td_toy_k1_ar" in src
    assert "procedure :: get_v0 => td_toy_k1_v0" in src
    assert "v0 = sum(n*self%a)/sum(n)" in src
    assert "real(pr), optional, intent(out) :: Arn2(size(n), size(n))" in src
    flat = src.replace("&\n", "")
    # one case per workload: mask = sum of the output bits
    # (Ar 1, ArV 2, Arn 64, all 1023)
    cases = dict(re.findall(r"case \((\d+)\)  ! (\w+):", flat))
    assert cases == {"1": "W0", "3": "W1", "65": "W3", "1023": "W6"}
    assert _call_keywords(flat, "3") == {
        "nc": "size(n)",
        "V": "v",
        "T": "t",
        "a": "self%a",
        "kij": "self%kij",
        "n": "n",
        "f": "Ar",
        "dfdV": "ArV",
    }
    w3 = re.search(r"case \(65\).*?\n\s*(call [^\n]*)", flat).group(1)
    assert "f=Ar" in w3 and "dfdn=Arn" in w3 and "dfdV" not in w3
    # default branch: the complete kernel (W6) through temporaries
    assert re.search(
        r"case default\s+call td_toy_k1_full\(self, n, v, t, Ar,", flat
    )
    full = flat[flat.index("subroutine td_toy_k1_full") :]
    assert (
        "call toy_k1_w6(" in full
        and "f=Ar_" in full
        and "dfdndn=Arn2_" in full
    )
    assert "if (present(ArTn)) ArTn = ArTn_" in full
    assert "real(pr) :: Arn2_(size(n), size(n))" in full
    assert src.count("merge(") == 10


def test_ge_wrapper_and_shared_kernel():
    # a kernel with optional outputs serving two workloads (as e2w)
    args = [
        ("nc", 0),
        ("T", 0),
        ("q", 1),
        ("n", 1),
        ("f", 0),
        ("dfdT", 0),
        ("dfdT2", 0),
        ("dfdn", 1),
        ("dfdndT", 1),
        ("dfdndn", 2),
    ]
    wls = {
        "G0": dict(routine="toy_g", args=args, outputs=["f"]),
        "G4": dict(
            routine="toy_g",
            args=args,
            outputs=["f", "dfdT", "dfdT2", "dfdn", "dfdndT", "dfdndn"],
        ),
    }
    src = yaeos_model_module("toy", "k2", "ge", wls, "td_toy_k2")
    assert "type, extends(GeModel) :: TD_TOY_K2" in src
    assert "procedure :: excess_gibbs => td_toy_k2_ge" in src
    assert "get_v0" not in src
    assert re.search(r"use td_toy_k2,\s*only:\s*toy_g\b", src)
    flat = src.replace("&\n", "")
    assert re.search(r"case \(1\)  ! G0: Ge\s", flat)
    assert _call_keywords(flat, "1") == {
        "nc": "size(n)",
        "T": "t",
        "q": "self%q",
        "n": "n",
        "f": "Ge",
    }
    assert "case (63)  ! G4" in flat
    signature = re.search(r"subroutine td_toy_k2_ge\((.*?)\)", flat).group(1)
    assert [name.strip() for name in signature.split(",")] == [
        "self",
        "n",
        "t",
        "Ge",
        "GeT",
        "GeT2",
        "Gen",
        "GeTn",
        "Gen2",
    ]


def test_errors():
    wls = _ar_workloads()
    with pytest.raises(ValueError, match="v0_expr"):
        yaeos_model_module("toy", "k1", "ar", wls, "m")
    del wls["W6"]
    with pytest.raises(ValueError, match="every output"):
        yaeos_model_module("toy", "k1", "ar", wls, "m", v0_expr="0")
    wls = _ar_workloads()
    wls["W1b"] = dict(wls["W1"])
    with pytest.raises(ValueError, match="same outputs"):
        yaeos_model_module("toy", "k1", "ar", wls, "m", v0_expr="0")


def test_foreign_output_is_not_a_parameter():
    # dfdV2 is an Ar output; in a Ge kernel it must not become a parameter
    args = [("nc", 0), ("T", 0), ("q", 1), ("n", 1), ("f", 0), ("dfdV2", 0)]
    wls = {"G0": dict(routine="toy_g", args=args, outputs=["f"])}
    with pytest.raises(ValueError, match=r"dfdV2.*'ge'"):
        yaeos_model_module("toy", "k2", "ge", wls, "td_toy_k2")


GE_STUB = """module yaeos__constants
   integer, parameter :: pr = 8
end module yaeos__constants

module yaeos__models_ge
   use yaeos__constants, only: pr
   implicit none
   type, abstract :: GeModel
   contains
      procedure(ge_interface), deferred :: excess_gibbs
   end type GeModel
   abstract interface
      subroutine ge_interface(self, n, t, Ge, GeT, GeT2, Gen, GeTn, Gen2)
         import GeModel, pr
         class(GeModel), intent(in) :: self
         real(pr), intent(in) :: n(:), t
         real(pr), optional, intent(out) :: Ge, GeT, GeT2
         real(pr), optional, intent(out) :: Gen(size(n)), GeTn(size(n))
         real(pr), optional, intent(out) :: Gen2(size(n), size(n))
      end subroutine ge_interface
   end interface
end module yaeos__models_ge
"""

GE_KERNELS = """module toy_kernels_k3
   implicit none
contains
   subroutine toy_full(nc, T, q, n, f, dfdT, dfdT2, dfdn, dfdndT, dfdndn)
      integer, intent(in) :: nc
      real(8), intent(in) :: T, q(:), n(:)
      real(8), intent(out) :: f, dfdT, dfdT2
      real(8), intent(out) :: dfdn(:), dfdndT(:), dfdndn(:, :)
      f = T * sum(n * q)
      dfdT = sum(n * q)
      dfdT2 = 0
      dfdn = T * q
      dfdndT = q
      dfdndn = 0
   end subroutine toy_full

   subroutine toy_lng(nc, T, q, n, f, dfdn)
      integer, intent(in) :: nc
      real(8), intent(in) :: T, q(:), n(:)
      real(8), intent(out) :: f
      real(8), intent(out) :: dfdn(:)
      f = T * sum(n * q)
      dfdn = T * q
   end subroutine toy_lng
end module toy_kernels_k3
"""

GE_DRIVER = """program drive
   use yaeos__constants, only: pr
   use td_toy_k3_model, only: TD_TOY_K3
   implicit none
   type(TD_TOY_K3) :: model
   real(pr) :: n(3), g_full, gen_full(3), gen_lng(3)
   real(pr) :: gt, gt2, gtn(3), gen2(3, 3)
   model%q = [1._pr, 2._pr, 3._pr]
   n = [0.2_pr, 0.3_pr, 0.5_pr]
   call model%excess_gibbs(n, 2._pr, g_full, gt, gt2, gen_full, gtn, gen2)
   call model%excess_gibbs(n, 2._pr, Gen=gen_lng)
   print "(a, 3es24.16)", "full", gen_full
   print "(a, 3es24.16)", "lng", gen_lng
end program drive
"""


@pytest.mark.slow
@pytest.mark.skipif(GFORTRAN is None, reason="gfortran not available")
def test_workload_without_f_compiles_and_matches_full(tmp_path):
    full = ["f", "dfdT", "dfdT2", "dfdn", "dfdndT", "dfdndn"]
    full_args = [("nc", 0), ("T", 0), ("q", 1), ("n", 1)] + [
        (o, {"dfdn": 1, "dfdndT": 1, "dfdndn": 2}.get(o, 0)) for o in full
    ]
    lng_args = full_args[:4] + [("f", 0), ("dfdn", 1)]
    wls = {
        "G_full": dict(routine="toy_full", args=full_args, outputs=full),
        "G_lng": dict(routine="toy_lng", args=lng_args, outputs=["dfdn"]),
    }
    src = yaeos_model_module("toy", "k3", "ge", wls, "toy_kernels_k3")
    assert "f_scratch_" in src
    files = {
        "stub.f90": GE_STUB,
        "kernels.f90": GE_KERNELS,
        "wrapper.f90": src,
        "driver.f90": GE_DRIVER,
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    result = compile_and_run(
        [tmp_path / name for name in files],
        tmp_path,
        ("-O0", "-ffree-line-length-0"),
    )
    values = {
        line.split()[0]: [float(x) for x in line.split()[1:]]
        for line in result.stdout.splitlines()
    }
    assert values["lng"] == values["full"] == [2.0, 4.0, 6.0]
