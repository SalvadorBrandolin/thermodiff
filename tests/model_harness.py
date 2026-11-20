"""Generic test harness (from the independent review of the chain-rule pass):
compile a .model with given settings, build an automatic gfortran driver that
calls it with nc = 3 and 7, and compare all outputs with the a0 reference."""

import sys, re, tempfile, pathlib, json, itertools, warnings

import numpy as np
from thermodiff.model import compile_model
from benchmarks.td_bench import EXTENDED_VARIANTS
from fortran_tools import DEBUG_FREE_FORM_FLAGS, compile_and_run

W6 = "dT,dV,dn,dT2,dV2,dndn,dTdV,dndT,dndV"

# Variants are applied with optional_derivatives set, so validate() is
# expected to report inert options; silence only that message here.
INERT_OPTIONS_MESSAGE = "options ignored by the current combination"

OUTS = [
    "f",
    "dfdT",
    "dfdV",
    "dfdn",
    "dfdT2",
    "dfdV2",
    "dfdndn",
    "dfdTdV",
    "dfdndT",
    "dfdndV",
]


def parse(code):
    m = re.search(
        r"subroutine (\w+)\((.*?)\)\n", code.replace("&\n        ", ""), re.S
    )
    name = m.group(1)
    args = [a.strip() for a in m.group(2).split(",")]
    ranks = {}
    for a in args:
        d = re.search(
            r"::\s*%s(\(([:, ]*)\))?\s*$" % re.escape(a), code, re.M | re.I
        )
        ranks[a] = 0 if not d or not d.group(1) else d.group(2).count(":")
    modname = re.search(r"^module (\w+)", code, re.M).group(1)
    return name, args, ranks, modname


def driver(code, inits, present, ncs=(3, 7)):
    name, args, ranks, mod = parse(code)
    ins = [a for a in args if a not in OUTS and a != "nc"]
    decl = []
    body = []
    for a in ins:
        r = ranks[a]
        if a in ("V", "T") or r == 0:
            decl.append(f"real(8) :: x_{a}")
            continue
        decl.append(f"real(8), allocatable :: x_{a}({','.join([':']*r)})")
    for o in OUTS:
        if o not in args:
            continue
        r = ranks[o]
        decl.append(
            f"real(8)"
            + (
                f", allocatable :: y_{o}({','.join([':']*r)})"
                if r
                else f" :: y_{o}"
            )
        )
    for nc in ncs:
        body.append(f"nc = {nc}")
        for a in ins:
            r = ranks[a]
            if a == "T":
                body.append(f"x_T = {inits.get('T','320d0')}")
                continue
            if a == "V":
                continue
            if r == 1:
                body.append(
                    f"allocate(x_{a}(nc)); do i=1,nc; x_{a}(i) = {inits.get(a, '0.5d0+0.13d0*i+0.02d0*i*i')}; end do"
                )
            elif r == 2:
                body.append(
                    f"allocate(x_{a}(nc,nc)); do i=1,nc; do j=1,nc; x_{a}(i,j) = {inits.get(a, '0.01d0*(i+j)+0.003d0*i*j')}; end do; end do"
                )
            elif r == 3:
                body.append(
                    f"allocate(x_{a}(nc,nc,nc)); do i=1,nc; do j=1,nc; do k=1,nc; x_{a}(i,j,k) = {inits.get(a, '0.002d0*(i+j+k)+0.0004d0*i*j*k')}; end do; end do; end do"
                )
            else:
                body.append(f"x_{a} = {inits.get(a,'0.7d0')}")
        if "V" in ins:
            body.append(f"x_V = {inits.get('V','0.9d0*sum(x_n)')}")
        for o in OUTS:
            if o not in args:
                continue
            r = ranks[o]
            if r == 1:
                body.append(f"allocate(y_{o}(nc)); y_{o} = -999d0")
            elif r == 2:
                body.append(f"allocate(y_{o}(nc,nc)); y_{o} = -999d0")
            else:
                body.append(f"y_{o} = -999d0")
        call = ", ".join(
            ["nc=nc"]
            + [f"{a}=x_{a}" for a in ins]
            + [f"{o}=y_{o}" for o in OUTS if o in args and o in present]
        )
        body.append(f"call {name}({call})")
        for o in OUTS:
            if o in args and o in present:
                body.append(f"write(u,'(es30.17e3)') y_{o}")
        for a in ins:
            if ranks[a] and a not in ("V", "T"):
                body.append(f"deallocate(x_{a})")
        for o in OUTS:
            if o in args and ranks[o]:
                body.append(f"deallocate(y_{o})")
    return (
        "module yaeos__constants\n implicit none\n real(8), parameter :: R = 0.0831446261815324d0\nend module\n"
        + code
        + f"\nprogram drive\n use {mod}\n implicit none\n integer :: nc,i,j,k,u\n"
        + "\n".join(decl)
        + "\nopen(newunit=u,file='out.txt')\n"
        + "\n".join(body)
        + "\nclose(u)\nend program\n"
    )


def run(text, settings, inits=None, present=OUTS, derivs=W6, opt=True):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=INERT_OPTIONS_MESSAGE, category=UserWarning
        )
        code = compile_model(
            text,
            derivatives=derivs,
            optional_derivatives=opt,
            **{"preset": "minimal", **settings},
        )
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        (d / "drive.f90").write_text(driver(code, inits or {}, present))
        compile_and_run([d / "drive.f90"], d, DEBUG_FREE_FORM_FLAGS)
        return np.loadtxt(d / "out.txt"), code


COMBOS = {k: EXTENDED_VARIANTS[k] for k in ("c3", "c6", "c6l", "c6f", "c6n")}
EXTRA = {
    "chain_only": dict(array_cse=False, share=False, chain_rule=True),
    "chain_psym": dict(
        array_cse=True, share=True, param_symmetry=True, chain_rule=True
    ),
    "chain_fuse_nohess": dict(
        array_cse=True,
        share=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
    ),
    "c6n_nohoist": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
    ),
}


def compare(text, inits=None, variants=None, present=OUTS, label=""):
    ref, _ = run(text, EXTENDED_VARIANTS["a0"], inits, present)
    res = {}
    for k, s in (variants or {**COMBOS, **EXTRA}).items():
        try:
            val, code = run(text, s, inits, present)
            rel = np.abs(val - ref) / np.maximum(np.abs(ref), 1e-8)
            res[k] = rel.max()
            uses_chain = "cmp" in code or "mat" in code
            print(
                f"{label} {k:18s} relerr {rel.max():.2e} argmax {rel.argmax()} n={len(ref)}"
            )
        except Exception as e:
            print(f"{label} {k:18s} ERROR {str(e)[:600]}")
    return res
