"""yaeos model wrappers for thermodiff-generated kernels.

A thermodiff variant of a model is a set of specialised kernels, one per
derivative workload (each computes exactly a fixed set of outputs).  This
module emits a Fortran module with a yaeos model type that owns the model
parameters (allocatable components) and implements the yaeos interface
(``ArModel%residual_helmholtz`` or ``GeModel%excess_gibbs``) by dispatching on
the set of optional outputs that are present:

* an integer mask is built from ``present()`` of every optional output
  (``merge``/sum, one bit per output in the order of the yaeos interface);
* a ``select case`` on the mask calls, for every workload, the kernel that
  computes exactly that output set, passing ``self%<parameter>`` and the
  present dummies directly (no copies);
* ``case default`` (any other combination, never used by the benchmark)
  calls the most complete kernel through a helper with local temporaries and
  copies the present outputs.

The emitted code is independent of the benchmark: it only needs the kernel
routine names, their argument lists (name and rank) and the output set of each
workload.  ``kernel_interface`` extracts the argument list from a generated
kernel source.
"""

import re

#: kernel output name -> yaeos dummy name, per family (order = yaeos interface order,
#: which also fixes the bit of each output in the mask)
AR_OUTPUTS = {
    "f": "Ar",
    "dfdV": "ArV",
    "dfdT": "ArT",
    "dfdTdV": "ArTV",
    "dfdV2": "ArV2",
    "dfdT2": "ArT2",
    "dfdn": "Arn",
    "dfdndV": "ArVn",
    "dfdndT": "ArTn",
    "dfdndn": "Arn2",
}
GE_OUTPUTS = {
    "f": "Ge",
    "dfdT": "GeT",
    "dfdT2": "GeT2",
    "dfdn": "Gen",
    "dfdndT": "GeTn",
    "dfdndn": "Gen2",
}
#: rank of every yaeos output (0 scalar, 1 size(n), 2 size(n) x size(n))
OUTPUT_RANK = {
    "Ar": 0,
    "ArV": 0,
    "ArT": 0,
    "ArTV": 0,
    "ArV2": 0,
    "ArT2": 0,
    "Arn": 1,
    "ArVn": 1,
    "ArTn": 1,
    "Arn2": 2,
    "Ge": 0,
    "GeT": 0,
    "GeT2": 0,
    "Gen": 1,
    "GeTn": 1,
    "Gen2": 2,
}
#: state arguments of a kernel -> actual argument inside the wrapper
STATE = {"nc": "size(n)", "n": "n", "V": "v", "T": "t"}
#: names that cannot be parameters (components of the yaeos base types, dummies, locals)
RESERVED = {"components", "name", "self", "mask", "v0", "p"} | {
    k.lower() for k in OUTPUT_RANK
}

_DECL = re.compile(
    r"^\s*(?:real|integer)\s*(?:\([^)]*\))?\s*(?:,[^:]*)?::\s*(.+)$", re.I
)


def family_outputs(family):
    if family == "ar":
        return AR_OUTPUTS
    if family == "ge":
        return GE_OUTPUTS
    raise ValueError(f"unknown family {family!r} (ar or ge)")


def kernel_interface(source):
    """Return ``(routine, [(argument, rank), ...])`` of a generated kernel.

    The rank comes from the declaration of each dummy argument (``x(:, :)``
    has rank 2); arguments in the argument list keep their order and spelling.
    """
    flat = source.replace("&\n", " ")
    match = re.search(r"subroutine\s+(\w+)\s*\(([^)]*)\)", flat, re.S)
    if not match:
        raise ValueError("no subroutine found in kernel source")
    routine = match.group(1)
    args = [
        a.strip()
        for a in match.group(2).replace("&", " ").split(",")
        if a.strip()
    ]
    ranks = {}
    for line in flat.splitlines():
        decl = _DECL.match(line)
        if not decl:
            continue
        for item in re.split(r",(?![^()]*\))", decl.group(1)):
            item = item.strip()
            name = re.match(r"(\w+)", item)
            if not name:
                continue
            dims = re.match(r"\w+\s*\(([^)]*)\)", item)
            ranks.setdefault(
                name.group(1).lower(),
                dims.group(1).count(",") + 1 if dims else 0,
            )
    missing = [a for a in args if a.lower() not in ranks]
    if missing:
        raise ValueError(f"{routine}: no declaration for {missing}")
    return routine, [(a, ranks[a.lower()]) for a in args]


def _wrap(line, indent, width=110):
    out, cur = [], indent + line
    while len(cur) > width:
        cut = cur.rfind(", ", 0, width)
        if cut < 0:
            break
        out.append(cur[: cut + 1] + " &")
        cur = indent + "    " + cur[cut + 2 :]
    out.append(cur)
    return out


def _call(routine, args, outs, actual):
    """Keyword call of a kernel; ``outs`` maps kernel output name -> actual argument."""
    parts = []
    for name, _ in args:
        if name in outs:
            parts.append(f"{name}={outs[name]}")
        elif name in actual:
            parts.append(f"{name}={actual[name]}")
    return f"call {routine}(" + ", ".join(parts) + ")"


def yaeos_model_module(
    model,
    key,
    family,
    workloads,
    kernel_module,
    *,
    v0_expr=None,
    type_name=None,
    module_name=None,
):
    """Fortran source of the yaeos wrapper of one thermodiff variant.

    Parameters
    ----------
    model, key:
        Fortran-safe model name (``pr76``, ``pcsaft_nd``) and variant key (``p33``);
        the type is ``TD_<MODEL>_<KEY>`` and the module ``td_<model>_<key>_model``
        unless ``type_name``/``module_name`` are given.
    family:
        ``"ar"`` (``extends(ArModel)``) or ``"ge"`` (``extends(GeModel)``).
    workloads:
        ordered mapping workload id -> ``dict(routine=..., args=[(name, rank)...],
        outputs=[kernel output names])``; ``args`` is the whole argument list of the
        kernel (as returned by :func:`kernel_interface`), ``outputs`` the outputs the
        workload requests (a kernel with optional outputs may serve several workloads).
    kernel_module:
        module that contains the kernels.
    v0_expr:
        Fortran expression of ``get_v0`` (Ar family, required there), in terms of
        ``n``, ``p``, ``t``, ``R`` (yaeos gas constant) and ``self%<parameter>``.
    """
    outmap = family_outputs(family)
    yaeos_names = list(outmap.values())
    bit = {nm: 1 << k for k, nm in enumerate(yaeos_names)}
    tname = type_name or f"TD_{model.upper()}_{key.upper()}"
    mname = module_name or f"td_{model.lower()}_{key.lower()}_model"
    proc = f"td_{model.lower()}_{key.lower()}"
    if family == "ar" and not v0_expr:
        raise ValueError("v0_expr is required for the Ar family")

    # model parameters: every kernel argument that is neither state nor output
    params, masks, full = {}, {}, None
    state = {s.lower() for s in STATE}
    for wid, wl in workloads.items():
        for name, rank in wl["args"]:
            low = name.lower()
            if low in state or name in outmap:
                continue
            if low in RESERVED:
                raise ValueError(
                    f"parameter name {name} clashes with the yaeos wrapper"
                )
            if params.setdefault(low, rank) != rank:
                raise ValueError(
                    f"parameter {name} has different ranks in different kernels"
                )
        for o in wl["outputs"]:
            if o not in outmap:
                raise ValueError(f"{wid}: unknown output {o}")
            if o not in {a for a, _ in wl["args"]}:
                raise ValueError(
                    f"{wid}: kernel {wl['routine']} has no output {o}"
                )
        mask = sum(bit[outmap[o]] for o in wl["outputs"])
        if mask in masks:
            raise ValueError(
                f"workloads {masks[mask]} and {wid} request the same outputs"
            )
        masks[mask] = wid
        if set(wl["outputs"]) == set(outmap):
            full = wid
    if full is None:
        raise ValueError(
            "no workload computes every output (needed for the default branch)"
        )

    def actual(prefix):
        act = {
            nm: f"{prefix}{nm.lower()}"
            for wl in workloads.values()
            for nm, _ in wl["args"]
            if nm.lower() in params
        }
        act.update({nm: STATE[nm] for nm in STATE})
        act.update({nm.lower(): STATE[nm] for nm in STATE})
        return act

    routines = list(dict.fromkeys(wl["routine"] for wl in workloads.values()))
    base, use_base = (
        ("ArModel", "yaeos__models_ar")
        if family == "ar"
        else ("GeModel", "yaeos__models_ge")
    )
    consts = "pr, R" if family == "ar" else "pr"
    L = [
        f"module {mname}",
        f"   !! yaeos {base} wrapper of the thermodiff kernels of {model} {key}.",
        "   !! Generated by thermodiff.yaeos_model -- do not edit.",
        "   !! residual dispatch: mask of present() outputs, one bit per output in the"
        " order of the yaeos interface;",
        "   !! every workload mask calls its kernel with the dummies directly, any other"
        " combination goes through the",
        f"   !! most complete kernel ({full}) with temporaries.",
        f"   use yaeos__constants, only: {consts}",
        f"   use {use_base}, only: {base}",
    ]
    L += _wrap(f"use {kernel_module}, only: " + ", ".join(routines), "   ")
    L += [
        "   implicit none",
        "   private",
        f"   public :: {tname}",
        "",
        f"   type, extends({base}) :: {tname}",
    ]
    for p, rank in params.items():
        dims = "" if rank == 0 else "(" + ", ".join([":"] * rank) + ")"
        L.append(f"      real(pr), allocatable :: {p}{dims}")
    L.append("   contains")
    if family == "ar":
        L += [
            f"      procedure :: residual_helmholtz => {proc}_ar",
            f"      procedure :: get_v0 => {proc}_v0",
        ]
    else:
        L += [f"      procedure :: excess_gibbs => {proc}_ge"]
    L += [f"   end type {tname}", "", "contains", ""]

    sig = ", ".join(yaeos_names)
    if family == "ar":
        head = f"subroutine {proc}_ar(self, n, v, t, {sig})"
        state_decl = ["real(pr), intent(in) :: n(:), v, t"]
        state_args = "n, v, t"
    else:
        head = f"subroutine {proc}_ge(self, n, t, {sig})"
        state_decl = ["real(pr), intent(in) :: n(:), t"]
        state_args = "n, t"

    def out_decls(optional=True):
        opt = ", optional" if optional else ""
        dl = []
        for nm in yaeos_names:
            r = OUTPUT_RANK[nm]
            dims = (
                ""
                if r == 0
                else "(size(n))" if r == 1 else "(size(n), size(n))"
            )
            dl.append(f"real(pr){opt}, intent(out) :: {nm}{dims}")
        return dl

    act = actual("self%")
    ind = "      "
    L += _wrap(head, "   ")
    L += [f"{ind}class({tname}), intent(in) :: self"] + [
        ind + d for d in state_decl
    ]
    L += [ind + d for d in out_decls()]
    L += [f"{ind}integer :: mask"]
    terms = [f"merge({bit[nm]}, 0, present({nm}))" for nm in yaeos_names]
    L += _wrap("mask = " + " + ".join(terms), ind)
    L += [f"{ind}select case (mask)"]
    for mask in sorted(masks):
        wid = masks[mask]
        wl = workloads[wid]
        present = [outmap[o] for o in wl["outputs"]]
        L.append(f"{ind}case ({mask})  ! {wid}: {', '.join(present)}")
        outs = {o: outmap[o] for o in wl["outputs"]}
        L += _wrap(_call(wl["routine"], wl["args"], outs, act), ind + "   ")
    L += [
        f"{ind}case default",
        f"{ind}   call {proc}_full(self, {state_args}, {sig})",
        f"{ind}end select",
        f"   end subroutine {proc}_{family}",
        "",
    ]

    # default branch: most complete kernel into temporaries, copy the present outputs
    fw = workloads[full]
    L += _wrap(f"subroutine {proc}_full(self, {state_args}, {sig})", "   ")
    L += [f"{ind}!! Any output combination: workload {full} into temporaries."]
    L += [f"{ind}class({tname}), intent(in) :: self"] + [
        ind + d for d in state_decl
    ]
    L += [ind + d for d in out_decls()]
    for nm in yaeos_names:
        r = OUTPUT_RANK[nm]
        dims = (
            "" if r == 0 else "(size(n))" if r == 1 else "(size(n), size(n))"
        )
        L.append(f"{ind}real(pr) :: {nm}_{dims}")
    outs = {o: outmap[o] + "_" for o in fw["outputs"]}
    L += _wrap(_call(fw["routine"], fw["args"], outs, act), ind)
    for nm in yaeos_names:
        L.append(f"{ind}if (present({nm})) {nm} = {nm}_")
    L += [f"   end subroutine {proc}_full", ""]

    if family == "ar":
        L += [
            f"   function {proc}_v0(self, n, p, t) result(v0)",
            f"{ind}!! Initial liquid volume for the yaeos volume solver (not timed).",
            f"{ind}class({tname}), intent(in) :: self",
            f"{ind}real(pr), intent(in) :: n(:), p, t",
            f"{ind}real(pr) :: v0",
        ]
        L += _wrap(f"v0 = {v0_expr}", ind)
        L += [f"   end function {proc}_v0", ""]
    L += [f"end module {mname}", ""]
    return "\n".join(L)
