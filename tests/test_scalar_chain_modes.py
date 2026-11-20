"""Pin the ``scalar_chain`` ``"estimate"``, ``"cost"`` and ``True`` paths.

The golden corpus builds with ``optional_derivatives=True``, where the
option is coerced off, so these cases (``optional_derivatives=False``) are the
only guard of the generated code and of the ``stats`` keys of those modes.
The pins were computed with the code before the re-entry was removed.  Each
entry maps ``(model, workload, variant, scalar_chain override)`` to the
SHA-256 of the code, the public stats and whether ``stats`` carries the
internal ``_structured`` entry (the known leak when ``scalar_chain`` is
``True``).
"""

import hashlib
import warnings
from pathlib import Path

import pytest

from thermodiff import bench_hooks
from thermodiff.fortran import to_fortran
from thermodiff.model.emit import build_diff
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "pr": ROOT / "examples" / "pr.model",
    "srk": ROOT / "tests" / "data" / "models" / "srk" / "srk.model",
    "uniquac": ROOT
    / "tests"
    / "data"
    / "models"
    / "uniquac"
    / "uniquac.model",
}
NON_OPTION_KEYS = ("key", "id", "token", "label", "legacy")
LEAK_KEY = "_structured"
HEAVY_MODELS = ("srk",)

PINS = {
    ("pr", "w3", "p30", None): (
        "865ecdee2441023efb5c0262150b9dd753398f5b89b9a82e40d452c78b521d04",
        {
            "n_temporaries": 12,
            "operation_count": 44,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_used": True,
            "scalar_operation_count": 22,
        },
        False,
    ),
    ("pr", "w3", "p32", None): (
        "8bff4ea033aabd19cce6af2651180434d417dc0ef6a1e0fedfc3a56e139fb3b7",
        {
            "n_temporaries": 12,
            "operation_count": 40,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_estimate": (20, 18),
            "scalar_chain_used": True,
            "scalar_operation_count": 22,
        },
        True,
    ),
    ("pr", "w3", "p33", None): (
        "8bff4ea033aabd19cce6af2651180434d417dc0ef6a1e0fedfc3a56e139fb3b7",
        {
            "n_temporaries": 12,
            "operation_count": 40,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_estimate": (20, 18),
            "scalar_chain_used": True,
            "scalar_operation_count": 22,
        },
        True,
    ),
    ("pr", "w3", "p32", True): (
        "8bff4ea033aabd19cce6af2651180434d417dc0ef6a1e0fedfc3a56e139fb3b7",
        {
            "n_temporaries": 12,
            "operation_count": 40,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_operation_count": 22,
        },
        True,
    ),
    ("pr", "w6", "p30", None): (
        "cadccd18b3a2c2e9638881086e42ef6497f286f549a06d0248a3784c56816d45",
        {
            "n_temporaries": 38,
            "operation_count": 149,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_used": True,
            "scalar_operation_count": 84,
        },
        False,
    ),
    ("pr", "w6", "p32", None): (
        "cadccd18b3a2c2e9638881086e42ef6497f286f549a06d0248a3784c56816d45",
        {
            "n_temporaries": 38,
            "operation_count": 149,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_estimate": (59, 56),
            "scalar_chain_used": True,
            "scalar_operation_count": 84,
        },
        True,
    ),
    ("pr", "w6", "p33", None): (
        "cadccd18b3a2c2e9638881086e42ef6497f286f549a06d0248a3784c56816d45",
        {
            "n_temporaries": 38,
            "operation_count": 149,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_estimate": (59, 56),
            "scalar_chain_used": True,
            "scalar_operation_count": 84,
        },
        True,
    ),
    ("pr", "w6", "p32", True): (
        "cadccd18b3a2c2e9638881086e42ef6497f286f549a06d0248a3784c56816d45",
        {
            "n_temporaries": 38,
            "operation_count": 149,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_operation_count": 84,
        },
        True,
    ),
    ("srk", "w3", "p30", None): (
        "e67fccc7c04593327bff101f9a2bc794b01e60f9f5e395f07f4fceaa6f7ca763",
        {
            "n_temporaries": 24,
            "operation_count": 70,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_used": True,
            "scalar_operation_count": 38,
        },
        False,
    ),
    ("srk", "w3", "p32", None): (
        "04ac3a517e7398d8298bcdef1c140391cccef13359320e507e88ed7a5b50d73e",
        {
            "n_temporaries": 24,
            "operation_count": 54,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_estimate": (39, 33),
            "scalar_chain_used": True,
            "scalar_operation_count": 38,
        },
        True,
    ),
    ("srk", "w3", "p33", None): (
        "04ac3a517e7398d8298bcdef1c140391cccef13359320e507e88ed7a5b50d73e",
        {
            "n_temporaries": 24,
            "operation_count": 54,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_estimate": (39, 33),
            "scalar_chain_used": True,
            "scalar_operation_count": 38,
        },
        True,
    ),
    ("srk", "w3", "p32", True): (
        "04ac3a517e7398d8298bcdef1c140391cccef13359320e507e88ed7a5b50d73e",
        {
            "n_temporaries": 24,
            "operation_count": 54,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_operation_count": 38,
        },
        True,
    ),
    ("srk", "w6", "p30", None): (
        "85f0dd71a6bf0f14720dd2e9be0d25e9dbd88973f563e57bc635a53c9da3f184",
        {
            "n_temporaries": 68,
            "operation_count": 215,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_used": True,
            "scalar_operation_count": 151,
        },
        False,
    ),
    ("srk", "w6", "p32", None): (
        "85f0dd71a6bf0f14720dd2e9be0d25e9dbd88973f563e57bc635a53c9da3f184",
        {
            "n_temporaries": 68,
            "operation_count": 215,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_estimate": (126, 118),
            "scalar_chain_used": True,
            "scalar_operation_count": 151,
        },
        True,
    ),
    ("srk", "w6", "p33", None): (
        "85f0dd71a6bf0f14720dd2e9be0d25e9dbd88973f563e57bc635a53c9da3f184",
        {
            "n_temporaries": 68,
            "operation_count": 215,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_estimate": (126, 118),
            "scalar_chain_used": True,
            "scalar_operation_count": 151,
        },
        True,
    ),
    ("srk", "w6", "p32", True): (
        "85f0dd71a6bf0f14720dd2e9be0d25e9dbd88973f563e57bc635a53c9da3f184",
        {
            "n_temporaries": 68,
            "operation_count": 215,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_operation_count": 151,
        },
        True,
    ),
    ("pr", "w0", "p32", None): (
        "84b1a5a3920e646e58326fa716edf04c2088a77be170f16131b950bd87faa6d4",
        {
            "n_temporaries": 3,
            "operation_count": 16,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_estimate": None,
            "scalar_chain_used": False,
            "scalar_operation_count": 13,
        },
        False,
    ),
    ("pr", "w2", "p30", None): (
        "c2f69e9807fcb7cd7bfa2b0ce23d47705f3acccd73a8f934062f6e51e7e1b0dc",
        {
            "n_temporaries": 7,
            "operation_count": 26,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_used": False,
            "scalar_operation_count": 21,
        },
        False,
    ),
    ("pr", "w0", "p30", None): (
        "84b1a5a3920e646e58326fa716edf04c2088a77be170f16131b950bd87faa6d4",
        {
            "n_temporaries": 3,
            "operation_count": 16,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_used": False,
            "scalar_operation_count": 13,
        },
        False,
    ),
    ("uniquac", "w3", "p30", None): (
        "8415bb960987cfe56a111191cd433708cbd1f6c5d48ff304ca6a644ce84052ea",
        {
            "n_temporaries": 14,
            "operation_count": 38,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_used": False,
            "scalar_operation_count": 13,
        },
        False,
    ),
    ("uniquac", "w3", "p32", None): (
        "2a4fec1b7e51ec687d4040202c86202697bea2e0a62c957c3ad067d605053ee5",
        {
            "n_temporaries": 17,
            "operation_count": 38,
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "scalar_chain_estimate": (7, 7),
            "scalar_chain_used": True,
            "scalar_operation_count": 14,
        },
        True,
    ),
}


def generate(model, workload, variant, scalar_chain):
    """Return ``(code, stats)`` of one case without optional derivatives."""
    path = MODELS[model]
    ir = build_ir(
        read_sections(path.read_text(encoding="utf-8"), str(path)), str(path)
    )
    derivatives = bench_hooks.WORKLOADS[workload]
    diff, _ = build_diff(ir, derivatives=derivatives)
    legacy = {
        key: value
        for key, value in bench_hooks.variant_settings(variant).items()
        if key not in NON_OPTION_KEYS
    }
    legacy["optional_derivatives"] = False
    if scalar_chain is not None:
        legacy["scalar_chain"] = scalar_chain
    stats = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = to_fortran(
            diff,
            derivatives=derivatives,
            name="f",
            subroutine_name="routine",
            integer_args=ir.dimensions,
            module_symbols=ir.imported,
            symmetric_params=getattr(ir, "symmetric", ()),
            stats=stats,
            **legacy,
        )
    return code, stats


def pin_params():
    """Parametrise the pins; the larger models are marked ``slow``."""
    params = []
    for key in PINS:
        marks = [pytest.mark.slow] if key[0] in HEAVY_MODELS else []
        case_id = "-".join(str(part) for part in key)
        params.append(pytest.param(*key, marks=marks, id=case_id))
    return params


@pytest.mark.parametrize(
    "model, workload, variant, scalar_chain", pin_params()
)
def test_scalar_chain_output_and_stats_are_pinned(
    model, workload, variant, scalar_chain
):
    digest, expected_stats, leaks = PINS[
        (model, workload, variant, scalar_chain)
    ]
    code, stats = generate(model, workload, variant, scalar_chain)
    assert (LEAK_KEY in stats) == leaks
    public = {k: v for k, v in stats.items() if k != LEAK_KEY}
    assert public == expected_stats
    assert hashlib.sha256(code.encode()).hexdigest() == digest
