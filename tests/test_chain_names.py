"""Pin how ``chain.py`` chooses slot and array names, and its output."""

import hashlib
import warnings
from pathlib import Path

import pytest
import sympy as sp

from benchmarks import td_bench as bench_hooks
from thermodiff import chain
from thermodiff.model.emit import build_diff
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections
from thermodiff.fortran import to_fortran

ROOT = Path(__file__).resolve().parent.parent
PR_MODEL = ROOT / "examples" / "pr.model"
NON_OPTION_KEYS = ("key", "id", "token", "label", "legacy")
PR76_MODEL = ROOT / "tests" / "data" / "models" / "pr76" / "pr76.model"
PR76_C6N_SHA256 = (
    "b26617ae2267e529521895b325f54971604e6b17bae918e2d430b56d93132b63"
)
FORCED_C6N_SHA256 = (
    "26fd8279ef1074b5de54e2d67d5febb19f1f63a189c85af9e90b2cfed9e84c6e"
)
PR_W6_E8_SHA256 = (
    "1743f775765989390d9509020938a0f152a2a8b8072fdf5b7a18c0e1836e03a4"
)


def configured_names(expression):
    """Return ``(component slot, pair slots, prefixes)`` chosen."""
    scheme = chain._name_scheme(expression)
    return (
        str(scheme.component_slot),
        tuple(str(slot) for slot in scheme.pair_slots),
        dict(scheme.prefixes),
    )


def test_free_expression_uses_default_names():
    expression = sp.Symbol("x") + 1
    assert configured_names(expression) == (
        "ic",
        ("ia", "ib"),
        {"cmp": "cmp", "mat": "mat"},
    )


def test_used_slot_names_are_skipped():
    expression = sp.Symbol("ic") + sp.Idx("ia") + sp.IndexedBase("IB")[1]
    component, pairs, _ = configured_names(expression)
    assert (component, pairs) == ("jc", ("ja", "jb"))


def test_nc_and_res_are_reserved():
    component, pairs, _ = configured_names(sp.Symbol("nc"))
    assert (component, pairs) == ("ic", ("ia", "ib"))


def test_prefix_clash_selects_next_candidate():
    expression = sp.Symbol("cmpx") + sp.Symbol("Matrix")
    _, _, prefixes = configured_names(expression)
    assert prefixes == {"cmp": "cpf", "mat": "pmt"}


def test_prefix_clashes_chain_through_candidates():
    expression = sp.Symbol("cmp1") + sp.Symbol("cpf1") + sp.Symbol("mat1")
    _, _, prefixes = configured_names(expression)
    assert prefixes == {"cmp": "cfn", "mat": "pmt"}


def generate(model_text, workload, variant):
    """Generate one case with ``optional_derivatives=False``."""
    ir = build_ir(read_sections(model_text, "model"), "model")
    derivatives = bench_hooks.WORKLOADS[workload]
    diff, _ = build_diff(ir, derivatives=derivatives)
    legacy = {
        key: value
        for key, value in bench_hooks.variant_settings(variant).items()
        if key not in NON_OPTION_KEYS
    }
    legacy["optional_derivatives"] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return to_fortran(
            diff,
            derivatives=derivatives,
            name="f",
            subroutine_name="routine",
            integer_args=ir.dimensions,
            module_symbols=ir.imported,
            symmetric_params=getattr(ir, "symmetric", ()),
            **legacy,
        )


def sha256(text):
    """Return the SHA-256 hex digest of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def forced_names_model():
    """PR76 with parameters named after slots and array prefixes."""
    return (
        PR76_MODEL.read_text(encoding="utf-8")
        .replace("    w     nc", "    ic    nc")
        .replace("w_i", "ic_i")
        .replace("Tc", "cmpTc")
        .replace("Pc", "matPc")
    )


def test_pr_w6_e8_chain_output_is_pinned():
    code = generate(PR_MODEL.read_text(encoding="utf-8"), "w6", "e8")
    assert sha256(code) == PR_W6_E8_SHA256


def test_component_arrays_output_is_pinned():
    model = PR76_MODEL.read_text(encoding="utf-8")
    assert sha256(generate(model, "w3", "c6n")) == PR76_C6N_SHA256


def test_forced_names_output_is_pinned():
    code = generate(forced_names_model(), "w3", "c6n")
    assert sha256(code) == FORCED_C6N_SHA256
    assert "jc" in code and "cpf" in code and "pmt" in code


@pytest.fixture(scope="module")
def isolated_codes():
    """Default and forced-names outputs, each generated alone."""
    default = generate(PR76_MODEL.read_text(encoding="utf-8"), "w3", "c6n")
    forced = generate(forced_names_model(), "w3", "c6n")
    return default, forced


@pytest.mark.parametrize("forced_first", [True, False])
def test_name_schemes_do_not_leak_across_calls(isolated_codes, forced_first):
    default_model = PR76_MODEL.read_text(encoding="utf-8")
    forced_model = forced_names_model()
    if forced_first:
        forced = generate(forced_model, "w3", "c6n")
        default = generate(default_model, "w3", "c6n")
    else:
        default = generate(default_model, "w3", "c6n")
        forced = generate(forced_model, "w3", "c6n")
    assert (default, forced) == isolated_codes


def test_interleaved_generations_match_isolated_ones(isolated_codes):
    default_model = PR76_MODEL.read_text(encoding="utf-8")
    forced_model = forced_names_model()
    codes = [
        generate(model, "w3", "c6n")
        for model in (
            forced_model,
            default_model,
            default_model,
            forced_model,
        )
    ]
    default, forced = isolated_codes
    assert codes == [forced, default, default, forced]
