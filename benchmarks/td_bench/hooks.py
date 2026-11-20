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
import warnings

from benchmarks.td_bench.registry import (
    EXTRA_KEYS,
    LEGACY_ALIASES,
    PROTOCOL_VARIANTS,
    STAGES,
    resolve_settings,
)
from thermodiff.codegen.options import CodegenOptions
from thermodiff.fortran import to_fortran
from thermodiff.model.emit import build_diff
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

__all__ = [
    "LEGACY_KEYS",
    "STAGE_VARIANTS",
    "VARIANTS",
    "WORKLOADS",
    "generate_workload",
    "variant_settings",
]

#: Stage registry indexed by key (see :mod:`benchmarks.td_bench.registry`).
_STAGES_BY_KEY = {entry.key: entry for entry in STAGES}

#: Optional codegen keys a variant may set (see ``to_fortran``).
_EXTRA_KEYS = EXTRA_KEYS

#: Full settings of every stage (a*, c*, it*), built once from the registry.
EXTENDED_VARIANTS = {
    key: resolve_settings(key, _STAGES_BY_KEY) for key in _STAGES_BY_KEY
}


def _protocol_entries():
    """Return the v1-v3 entries, then one per extended stage.

    ``key`` is the short name used in file and routine names
    (``pr76_v1_w0``), ``token`` the directory infix (``pr76_v1_none``) and
    ``id`` the method id of the protocol.
    """
    entries = {
        key: dict(
            key=key,
            id=method_id,
            token=token,
            label=label,
            **EXTENDED_VARIANTS[stage_key],
        )
        for key, method_id, token, label, stage_key in PROTOCOL_VARIANTS
    }
    for key, settings in EXTENDED_VARIANTS.items():
        entries[key] = dict(
            key=key,
            id=f"td_{key}",
            token=key,
            label=key.upper(),
            **settings,
        )
    return entries


#: Benchmark generation variants (D-007), built once.
VARIANTS = _protocol_entries()

#: Standard names (NAMING.md) -> legacy keys, see the registry.
LEGACY_KEYS = dict(LEGACY_ALIASES)

STAGE_VARIANTS = {
    key: dict(
        key=key,
        id=f"td-{key}",
        token=key,
        label=key,
        legacy=old,
        **EXTENDED_VARIANTS[old],
    )
    for key, old in LEGACY_KEYS.items()
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
    # excess Gibbs models, variables (n, T)
    "g0": "none",
    "g1": "dn",
    "g2": "dT,dT2",
    "g3": "dn,dndn",
    "g4": "dT,dT2,dn,dndT,dndn",
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
    if key in STAGE_VARIANTS:
        return STAGE_VARIANTS[key]
    for entry in list(VARIANTS.values()) + list(STAGE_VARIANTS.values()):
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


def _normalised_options(settings, optional_derivatives):
    """Build the validated options of a variant, silencing the warning.

    The benchmark variants deliberately enable options that some
    combinations ignore, so the ``UserWarning`` of
    :meth:`CodegenOptions.validate` is expected noise here; the ignored
    names are returned instead so that they are still recorded.

    Parameters
    ----------
    settings : dict
        Variant settings as returned by :func:`variant_settings`.
    optional_derivatives : bool
        Declare every output as ``optional, intent(out)``.

    Returns
    -------
    tuple
        The normalised :class:`CodegenOptions` and the tuple of the names
        of the options that were ignored and disabled.
    """
    codegen_settings = {
        key: settings[key]
        for key in ("array_cse", "share", *_EXTRA_KEYS)
        if key in settings
    }
    options = CodegenOptions.from_legacy(
        **codegen_settings, optional_derivatives=optional_derivatives
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return options.validate()


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
        ``n_temporaries``, ``ignored_options`` (names of the options the
        combination ignored) and ``source`` (the emitted file content).
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

    options, ignored_options = _normalised_options(
        settings, optional_derivatives
    )
    stats = {"ignored_options": ignored_options}
    phase = time.perf_counter()
    code = to_fortran(
        diff,
        derivatives=WORKLOADS[workload],
        name="f",
        subroutine_name=routine,
        integer_args=ir.dimensions,
        module_symbols=ir.imported,
        options=options,
        stats=stats,
        symmetric_params=getattr(ir, "symmetric", ()),
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
        **{key: settings[key] for key in _EXTRA_KEYS if key in settings},
        "ignored_options": ignored_options,
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
        "scalar_chain_used": stats.get("scalar_chain_used"),
        "scalar_chain_estimate": stats.get("scalar_chain_estimate"),
        "scalar_operation_count": stats.get("scalar_operation_count"),
        "n_temporaries": stats.get("n_temporaries"),
        "source": code,
    }
