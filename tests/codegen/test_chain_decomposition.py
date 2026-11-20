"""Pins of the chain-rule structures behind ``chain.py``.

Each case compiles a model and records, through wrappers of
``chain_rule_derivatives`` and ``component_definitions``, the ``srepr`` hash
of every returned structure (keys and order included) and the hash of the
generated Fortran text.
"""

import hashlib
import re
from pathlib import Path

import pytest
import sympy as sp

from thermodiff import chain
from thermodiff.model import compile_model
from thermodiff.model.examples import EXAMPLES

MODELS_DIR = Path(__file__).resolve().parents[1] / "data" / "models"
DUMMY_INDEX = re.compile(r"dummy_index=\d+")
DIGEST_LENGTH = 16


def digest(text):
    """Return the short SHA-256 of ``text``."""
    return hashlib.sha256(text.encode()).hexdigest()[:DIGEST_LENGTH]


def stable_repr(value):
    """Return a ``srepr`` that does not depend on dummy counters.

    Parameters
    ----------
    value : object
        A SymPy object, or a list, tuple or dict of them.

    Returns
    -------
    str
        Text with the ``dummy_index`` numbers removed.
    """
    if isinstance(value, dict):
        return "{%s}" % ", ".join(
            f"{key!r}: {stable_repr(item)}" for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return "[%s]" % ", ".join(stable_repr(item) for item in value)
    if isinstance(value, (sp.Basic, int, float)):
        return DUMMY_INDEX.sub("dummy_index=0", sp.srepr(sp.sympify(value)))
    return repr(value)


def structure_digest(structured):
    """Return the digest of every value of a structured result."""
    if structured is None:
        return None
    return [
        (key, digest(stable_repr(value)))
        for key, value in structured.items()
        if key != chain.NAME_SCHEME_KEY
    ]


def definitions_digest(defs):
    """Return the digest of a ``component_definitions`` list."""
    return [
        (name, digest(stable_repr(body)), rank, symmetric)
        for name, body, rank, symmetric in defs
    ]


def run_case(text, options, monkeypatch):
    """Compile ``text`` and return the recorded digests."""
    recorded = {"chain": [], "defs": []}
    original_chain = chain.chain_rule_derivatives
    original_defs = chain.component_definitions

    def spy_chain(*args, **kwargs):
        result = original_chain(*args, **kwargs)
        recorded["chain"].append(structure_digest(result))
        return result

    def spy_defs(*args, **kwargs):
        defs, mapping = original_defs(*args, **kwargs)
        recorded["defs"].append(definitions_digest(defs))
        return defs, mapping

    monkeypatch.setattr(chain, "chain_rule_derivatives", spy_chain)
    monkeypatch.setattr(chain, "component_definitions", spy_defs)
    code = compile_model(text, optional_derivatives=False, **options)
    return recorded, digest(code)


def model_text(name):
    """Return the text of a test model or of an example."""
    if name == "pr":
        return EXAMPLES["Peng-Robinson (pressure)"]
    return (MODELS_DIR / name / f"{name}.model").read_text()


CASES = {
    "pr-standard": ("pr", {"preset": "standard"}),
    "pr-fastest": ("pr", {"preset": "fastest"}),
    "pr-scalar-chain": ("pr", {"preset": "fastest", "scalar_chain": True}),
    "cmr-standard": ("cmr", {"preset": "standard"}),
    "cmr-scalar-chain": ("cmr", {"preset": "fastest", "scalar_chain": True}),
    "uniquac-scalar-chain": (
        "uniquac",
        {"preset": "fastest", "scalar_chain": True},
    ),
    "pcsaft-standard": ("pcsaft", {"preset": "standard"}),
    "pcsaft-fastest": ("pcsaft", {"preset": "fastest"}),
}
HEAVY = ("uniquac-scalar-chain", "pcsaft-standard", "pcsaft-fastest")
#: ``label`` -> (digest of the recorded structures, digest of the code).
PINS = {
    "pr-standard": (
        "edd3676813783aee",
        "dabaa4e42240b70e",
    ),
    "pr-fastest": (
        "d6f1972b1d792df6",
        "25a8b3fbfb4552df",
    ),
    "pr-scalar-chain": (
        "d6f1972b1d792df6",
        "25a8b3fbfb4552df",
    ),
    "cmr-standard": (
        "7a9459dc12cefa07",
        "484ef4b6a0b80adb",
    ),
    "cmr-scalar-chain": (
        "78a1544f676df25d",
        "ac3a42baf2b39c72",
    ),
    "uniquac-scalar-chain": (
        "a5dab2883ef75cc3",
        "6b6fb31dd9328db5",
    ),
    "pcsaft-standard": (
        "52e9407ed9bcae83",
        "87a05500699a1a1b",
    ),
    "pcsaft-fastest": (
        "5c439fa56a3e6951",
        "ec14165e9b9c5b86",
    ),
}


def case_params():
    """Parametrise the cases; the heavy ones are marked ``slow``."""
    return [
        pytest.param(
            label,
            marks=[pytest.mark.slow] if label in HEAVY else [],
            id=label,
        )
        for label in CASES
    ]


@pytest.mark.parametrize("label", case_params())
def test_chain_structures_are_pinned(label, monkeypatch):
    model, options = CASES[label]
    recorded, code_digest = run_case(model_text(model), options, monkeypatch)
    assert (digest(repr(recorded)), code_digest) == PINS[label]
