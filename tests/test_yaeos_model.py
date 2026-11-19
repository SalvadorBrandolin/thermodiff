"""Tests of thermodiff.yaeos_model (yaeos ArModel/GeModel wrappers of the kernels).

Pure Python: the emitted Fortran source is checked textually for a small case
(type name, parameter components and ranks, one ``case`` per workload mask with
the right kernel and outputs, and the default branch).
"""

import re

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
    # one case per workload: mask = sum of the output bits (Ar 1, ArV 2, Arn 64, all 1023)
    cases = dict(re.findall(r"case \((\d+)\)  ! (\w+):", flat))
    assert cases == {"1": "W0", "3": "W1", "65": "W3", "1023": "W6"}
    w1 = re.search(r"case \(3\).*?\n\s*(call [^\n]*)", flat).group(1)
    assert re.sub(r"\s+", " ", w1) == (
        "call toy_k1_w1(nc=size(n), V=v, T=t, a=self%a, "
        "kij=self%kij, n=n, f=Ar, dfdV=ArV)"
    )
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
    assert "use td_toy_k2, only: toy_g\n" in src
    flat = src.replace("&\n", "")
    assert re.search(
        r"case \(1\)  ! G0: Ge\s+call toy_g\(nc=size\(n\), T=t, q=self%q, n=n, f=Ge\)",
        flat,
    )
    assert "case (63)  ! G4" in flat
    assert (
        "subroutine td_toy_k2_ge(self, n, t, Ge, GeT, GeT2, Gen, GeTn, Gen2)"
        in flat
    )


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
