"""Measurement hooks for the benchmark's generation phase.

This module exposes the callable API used by the session-root driver
(``scripts/generate-symbolic-variants.py``) and by ``benchmark-engineer``
to record, per (workload x variant):

* the wall time of generation and of its sub-phases (parse/IR,
  differentiation, Fortran emission),
* the path, line count and byte count of the generated file,
* the symbolic operation count of the final expressions (supplementary
  metadata; it is not a runtime predictor).

Every benchmark generation uses ``optional_derivatives=True`` (D-014), so
it is the default here.  The generated files are deterministic: identical
inputs give byte-identical output; no timestamps are ever written into a
generated file.  The ``use`` statements declared by the ``.model`` file are
spliced at generation time as the first body lines of the emitted bare
subroutine (so each file compiles standalone; see the interface-contract
handoff, producer side, route (a)).

Variant settings (D-007; ``reports/protocol.md`` section 13):

===========  ==========  =======  ==================================
variant      array_cse   share    passes run
===========  ==========  =======  ==================================
v1 "none"    False       False    none
v2 "partial" False       True     _share_sums + _common_scalar_cse
v3 "full"    True        True     _indexed_cse + the two above
===========  ==========  =======  ==================================
"""

import os
import time

from thermodiff.fortran import to_fortran
from thermodiff.model.emit import build_diff
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

__all__ = [
    "VARIANTS",
    "WORKLOADS",
    "generate_workload",
    "variant_settings",
]

#: Benchmark generation variants (D-007).  ``key`` is the short name used
#: in file and routine names (``pr76_v1_w0``), ``token`` the directory
#: infix (``pr76_v1_none``) and ``id`` the method id of the protocol.
VARIANTS = {
    "v1": {
        "key": "v1",
        "id": "td_v1_none",
        "token": "none",
        "label": "V1 none",
        "array_cse": False,
        "share": False,
    },
    "v2": {
        "key": "v2",
        "id": "td_v2_partial",
        "token": "partial",
        "label": "V2 partial",
        "array_cse": False,
        "share": True,
    },
    "v3": {
        "key": "v3",
        "id": "td_v3_full",
        "token": "full",
        "label": "V3 full",
        "array_cse": True,
        "share": True,
    },
}

#: Workload id -> ``derivatives=`` string (frozen interface contract).
WORKLOADS = {
    "w0": "none",
    "w1": "dV",
    "w2": "dT",
    "w3": "dn",
    "w4": "dT,dV,dn",
    "w5a": "dV,dV2",
    "w5b": "dT,dV,dT2,dTdV",
    "w5c": "dn,dndn",
    "w5d": "dn,dndV,dndT",
    "w6": "dT,dV,dn,dT2,dV2,dndn,dTdV,dndT,dndV",
}

#: Indentation used by the emitted bare subroutine.
_INDENT = "   "


def variant_settings(variant):
    """Return the settings dict of a generation variant.

    Parameters
    ----------
    variant : str
        ``"v1"``, ``"v2"``, ``"v3"``, or a protocol method id such as
        ``"td_v2_partial"``.

    Returns
    -------
    dict
        The matching entry of :data:`VARIANTS`.

    Raises
    ------
    ValueError
        If the variant is unknown.
    """
    key = str(variant).strip().lower()
    if key in VARIANTS:
        return VARIANTS[key]
    for entry in VARIANTS.values():
        if key == entry["id"]:
            return entry
    raise ValueError(
        f"unknown variant {variant!r}; expected one of "
        + ", ".join(sorted(VARIANTS))
    )


def _normalise_workload(workload):
    """Return the canonical workload id (lower case, e.g. ``w5a``)."""
    key = str(workload).strip().lower()
    if key.startswith("w") and key[1:] in WORKLOADS:
        key = "w" + key[1:]
    if key not in WORKLOADS:
        raise ValueError(
            f"unknown workload {workload!r}; expected one of "
            + ", ".join(WORKLOADS)
        )
    return key


def _splice_uses(code, uses):
    """Insert the model ``use`` lines before ``implicit none``.

    The splice is deterministic and happens at generation time, never at
    build time: the generated file is the final artifact.

    Parameters
    ----------
    code : str
        Emitted bare subroutine source.
    uses : list of str
        ``use`` clauses as written in the ``.model`` file.

    Returns
    -------
    str
        Source with the ``use`` lines spliced in.
    """
    if not uses:
        return code
    lines = code.splitlines()
    for position, line in enumerate(lines):
        if line.strip() == "implicit none":
            lines[position:position] = [
                f"{_INDENT}use {use_text}" for use_text in uses
            ]
            break
    else:  # pragma: no cover - to_fortran always emits implicit none
        raise RuntimeError("generated code has no 'implicit none' line")
    return "\n".join(lines) + "\n"


def generate_workload(
    model_path,
    workload,
    variant,
    out_dir,
    optional_derivatives=True,
    subroutine_name=None,
    write=True,
):
    """Generate one workload of one variant and measure the generation.

    Parameters
    ----------
    model_path : str
        Path to the ``.model`` file.
    workload : str
        Workload id (``w0`` ... ``w6``, ``w5a``, ``w5b``, ``w5c``,
        ``w5d``), case insensitive.
    variant : str
        ``v1``/``v2``/``v3`` or a protocol method id (see
        :func:`variant_settings`).
    out_dir : str
        Directory the generated ``.f90`` file is written into.
    optional_derivatives : bool, optional
        Declare every output as ``optional, intent(out)`` (D-014 default).
    subroutine_name : str, optional
        Name of the emitted subroutine.  Defaults to
        ``<model>_<variant key>_<workload>``, e.g. ``pr76_v1_w0``.
    write : bool, optional
        Write the ``.f90`` file (default).  With ``False`` the metadata is
        still returned, with ``path`` pointing at where the file would go.

    Returns
    -------
    dict
        Metadata with keys ``model``, ``variant``, ``variant_id``,
        ``variant_token``, ``workload``, ``derivatives``,
        ``subroutine_name``, ``optional_derivatives``, ``array_cse``,
        ``share``, ``spliced_uses``, ``path``, ``generation_time_s``,
        ``parse_time_s``, ``diff_time_s``, ``emit_time_s``, ``lines``,
        ``bytes``, ``operation_count``, ``operation_count_tool``,
        ``n_temporaries`` and ``source`` (the emitted file content).
    """
    workload = _normalise_workload(workload)
    settings = variant_settings(variant)

    with open(model_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    started = time.perf_counter()
    phase = time.perf_counter()
    raw = read_sections(text, model_path)
    ir = build_ir(raw, model_path)
    parse_time_s = time.perf_counter() - phase

    phase = time.perf_counter()
    diff, _selection = build_diff(ir, derivatives=WORKLOADS[workload])
    diff_time_s = time.perf_counter() - phase

    name = os.path.splitext(os.path.basename(str(model_path)))[0]
    model_name = ir.model_name or name
    routine = subroutine_name or f"{model_name}_{settings['key']}_{workload}"

    stats = {}
    phase = time.perf_counter()
    code = to_fortran(
        diff,
        derivatives=WORKLOADS[workload],
        name="f",
        subroutine_name=routine,
        integer_args=ir.dimensions,
        module_symbols=ir.imported,
        optional_derivatives=optional_derivatives,
        array_cse=settings["array_cse"],
        share=settings["share"],
        stats=stats,
    )
    emit_time_s = time.perf_counter() - phase

    code = _splice_uses(code, ir.uses)
    generation_time_s = time.perf_counter() - started

    path = os.path.join(out_dir, f"{routine}.f90")
    if write:
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(code)

    lines = code.count("\n")
    return {
        "model": model_name,
        "variant": settings["key"],
        "variant_id": settings["id"],
        "variant_token": settings["token"],
        "workload": workload,
        "derivatives": WORKLOADS[workload],
        "subroutine_name": routine,
        "optional_derivatives": bool(optional_derivatives),
        "array_cse": settings["array_cse"],
        "share": settings["share"],
        "spliced_uses": list(ir.uses),
        "path": path,
        "generation_time_s": generation_time_s,
        "parse_time_s": parse_time_s,
        "diff_time_s": diff_time_s,
        "emit_time_s": emit_time_s,
        "lines": lines,
        "bytes": len(code.encode("utf-8")),
        "operation_count": stats.get("operation_count"),
        "operation_count_tool": stats.get("operation_count_tool"),
        "n_temporaries": stats.get("n_temporaries"),
        "source": code,
    }
