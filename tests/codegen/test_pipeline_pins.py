"""Pin the full output of the emit pipeline for a few representative cases.

Each pin holds the SHA-256 of the Fortran text and of the temporaries the
namer created (name and order), so that restructuring ``_emit_code`` into
passes cannot change the generated code or the numbering of the ``tN``.
The cases run without optional derivatives so that every pass is active.
"""

import hashlib
import warnings
from pathlib import Path

import pytest

from thermodiff.codegen import CodegenOptions, pipeline
from thermodiff.fortran import to_fortran
from thermodiff.model.emit import build_diff
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

ROOT = Path(__file__).resolve().parent.parent.parent
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
WORKLOADS = {
    "w3": "dn",
    "w6": "dT,dV,dn,dT2,dV2,dndn,dTdV,dndT,dndV",
    "g4": "dT,dT2,dn,dndT,dndn",
}
#: Settings that are a preset with overrides (the benchmark stage variants
#: ``p32`` and ``p33`` are ``fastest`` with and without ``split_acc``).
OVERRIDES = {"fastest-no-split-acc": ("fastest", {"split_acc": False})}
PRESETS = ("minimal", "standard", "fastest")
HEAVY_MODELS = ("srk", "uniquac")

#: ``(model, workload, setting)`` -> (digest of the code, digest of the
#: temporaries created by the namer); ``setting`` is a preset or a variant.
PINS = {
    ("pr", "w6", "minimal"): (
        "c9b54765ef24b842bac4178821e797c7d15921eb01de94c810e465ba1484860e",
        "b52c934b9c624d5837a6220aab0406301073e8d24206786bce7fb19a236f1530",
    ),
    ("pr", "w6", "standard"): (
        "1743f775765989390d9509020938a0f152a2a8b8072fdf5b7a18c0e1836e03a4",
        "692e40ae6021e028f2c3856f91a3c84d3c2c435fa022c14e22025917afd4c4e7",
    ),
    ("pr", "w6", "fastest"): (
        "cadccd18b3a2c2e9638881086e42ef6497f286f549a06d0248a3784c56816d45",
        "1df761fb620415e10106a26cc5e65073f07471e41ce03c02438612a59cd9e76b",
    ),
    ("srk", "w3", "fastest"): (
        "04ac3a517e7398d8298bcdef1c140391cccef13359320e507e88ed7a5b50d73e",
        "e59aae2d12130ef5c60499bd8b3bd77188601e43891a57e8b33f3a3d43240564",
    ),
    ("uniquac", "g4", "fastest"): (
        "2a2a584df4da62bfcde27aa8d6e440e9cdcf66954614a83d7e2ff5e922edeb50",
        "7522af3de99d2dd421992cf5558ccea2c7e24d050c5f03553d350a233206e3ea",
    ),
    ("pr", "w3", "fastest-no-split-acc"): (
        "8bff4ea033aabd19cce6af2651180434d417dc0ef6a1e0fedfc3a56e139fb3b7",
        "34179cd2fc9acbce24a61147033a5029c741b892da63c0b3e28d3a16b76d8dad",
    ),
}


class RecordingNamer(pipeline._Namer):
    """Namer that remembers the instance of the last emission."""

    last = None

    def __init__(self):
        super().__init__()
        type(self).last = self


def build_options(setting):
    """Return the options of a preset or of a preset with overrides.

    Parameters
    ----------
    setting : str
        A preset name or a key of ``OVERRIDES``.

    Returns
    -------
    CodegenOptions
        Options with ``optional_derivatives`` off.
    """
    preset, overrides = OVERRIDES.get(setting, (setting, {}))
    return CodegenOptions.from_preset(
        preset, optional_derivatives=False, **overrides
    )


def generate(model, workload, setting, monkeypatch):
    """Return the digests of the code and of the created temporaries."""
    path = MODELS[model]
    ir = build_ir(
        read_sections(path.read_text(encoding="utf-8"), str(path)), str(path)
    )
    derivatives = WORKLOADS[workload]
    diff, _ = build_diff(ir, derivatives=derivatives)
    monkeypatch.setattr(pipeline, "_Namer", RecordingNamer)
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
            options=build_options(setting),
        )
    created = "\n".join(str(sym) for sym in RecordingNamer.last.created)
    return (
        hashlib.sha256(code.encode()).hexdigest(),
        hashlib.sha256(created.encode()).hexdigest(),
    )


def pin_params():
    """Parametrise the pins; the larger models are marked ``slow``."""
    params = []
    for key in PINS:
        marks = [pytest.mark.slow] if key[0] in HEAVY_MODELS else []
        params.append(pytest.param(*key, marks=marks, id="-".join(key)))
    return params


@pytest.mark.parametrize("model, workload, setting", pin_params())
def test_emit_output_and_temporaries_are_pinned(
    model, workload, setting, monkeypatch
):
    assert (
        generate(model, workload, setting, monkeypatch)
        == PINS[(model, workload, setting)]
    )
