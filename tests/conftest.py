"""Shared fixtures for the thermodiff test suite."""

import os

import pytest

from thermodiff.bench_hooks import generate_workload, variant_settings

#: Root of the test fixtures (``models/`` and ``scripts/``).
SESSION_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

#: Canonical PR76 model of the benchmark (protocol section 2.3).
PR76_MODEL = os.path.join(SESSION_ROOT, "models", "pr76", "pr76.model")

#: Frozen generation order of the ten workloads.
WORKLOAD_ORDER = [
    "w0",
    "w1",
    "w2",
    "w3",
    "w4",
    "w5a",
    "w5b",
    "w5c",
    "w5d",
    "w6",
]

#: Frozen variant order.
VARIANT_ORDER = ["v1", "v2", "v3"]


@pytest.fixture(scope="session")
def pr76_generated(tmp_path_factory):
    """Generate the 30 real PR76 (workload x variant) files once.

    Returns
    -------
    dict
        ``{"root": Path, "meta": {(variant, workload): hook metadata}}``
        with the files written below ``root`` in the frozen layout
        ``pr76_<variant>_<token>/pr76_<variant>_<workload>.f90``.
    """
    root = tmp_path_factory.mktemp("pr76_generated")
    meta = {}
    for variant in VARIANT_ORDER:
        settings = variant_settings(variant)
        out_dir = root / f"pr76_{variant}_{settings['token']}"
        for workload in WORKLOAD_ORDER:
            metadata = generate_workload(
                PR76_MODEL,
                workload,
                variant,
                str(out_dir),
            )
            meta[(variant, workload)] = metadata
    return {"root": root, "meta": meta}
