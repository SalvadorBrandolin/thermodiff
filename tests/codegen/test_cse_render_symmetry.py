"""Pins of ``_share_sums``, ``_render_outer_fused`` and dummy merging.

The three functions are pinned on small synthetic inputs (the digest of every
returned value, order included, plus the namer and arrays they fill) and, end
to end, through the hash of the Fortran text of models where they run.
"""

import hashlib
import re
from pathlib import Path

import pytest
import sympy as sp

from thermodiff.codegen import cse, render, symmetry
from thermodiff.codegen.common import _Namer
from thermodiff.codegen.ir import (
    ComponentFusedMap,
    OuterFusedNest,
    TempDef,
)
from thermodiff.model import compile_model
from thermodiff.model.examples import EXAMPLES
from thermodiff.thermovars import i, j, k, l, m, nc

MODELS_DIR = Path(__file__).resolve().parents[1] / "data" / "models"
DUMMY_INDEX = re.compile(r"dummy_index=\d+")
DIGEST_LENGTH = 16
NC = sp.Symbol("nc")
a_arr = sp.IndexedBase("a")
b_arr = sp.IndexedBase("b")
c_arr = sp.IndexedBase("c")
x_arr = sp.IndexedBase("x")
kij = sp.IndexedBase("kij")
lam = sp.IndexedBase("lam")
cmp0 = sp.IndexedBase("cmp0")
u_arr = sp.IndexedBase("u")


def digest(value):
    """Return the short SHA-256 of ``value`` without dummy counters."""
    text = DUMMY_INDEX.sub("dummy_index=0", stable_text(value))
    return hashlib.sha256(text.encode()).hexdigest()[:DIGEST_LENGTH]


def stable_text(value):
    """Return a deterministic text of nested containers and records."""
    if isinstance(value, TempDef):
        fields = vars(value)
        return "%s(%s)" % (
            type(value).__name__,
            ", ".join(
                f"{key}={stable_text(item)}" for key, item in fields.items()
            ),
        )
    if isinstance(value, dict):
        return "{%s}" % ", ".join(
            f"{stable_text(key)}: {stable_text(item)}"
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return "[%s]" % ", ".join(stable_text(item) for item in value)
    if isinstance(value, (sp.Basic, int, float)):
        return sp.srepr(sp.sympify(value))
    return repr(value)


# --- _share_sums ------------------------------------------------------


def total(term, *indices):
    """Return ``sum_indices term`` over ``1..nc``."""
    return sp.Sum(term, *[(index, 1, nc) for index in indices])


def share_cases():
    """Return ``label -> (expressions, options)`` for ``_share_sums``."""
    shared = total(a_arr[k] * b_arr[k], k)
    renamed = total(a_arr[l] * b_arr[l], l)
    row_k = total(x_arr[i, k] * a_arr[k], k)
    row_l = total(x_arr[j, l] * a_arr[l], l)
    pair_ij = total(x_arr[i, j, k] * a_arr[k], k)
    pair_ji = total(x_arr[j, i, k] * a_arr[k], k)
    symmetric_ij = total(kij[i, j] * a_arr[k], k)
    symmetric_ji = total(kij[j, i] * a_arr[k], k)
    nested = total(a_arr[k] * total(b_arr[k] * x_arr[k, l], l), k)
    return {
        "repeated": (
            [shared * 2, shared + c_arr[i], total(c_arr[k], k)],
            {},
        ),
        "alpha": (
            [shared + renamed, total(a_arr[m] * b_arr[m], m) * 3],
            {"alpha": True},
        ),
        "no-alpha": ([shared + renamed], {}),
        "nested": ([nested, nested * a_arr[i]], {}),
        "free-alpha": (
            [row_k * c_arr[i], row_l * c_arr[j], shared + renamed],
            {"free_alpha": True},
        ),
        "free-alpha-transpose": (
            [pair_ij * c_arr[i], pair_ji * c_arr[j]],
            {"free_alpha": True, "transpose": True},
        ),
        "free-alpha-no-transpose": (
            [pair_ij * c_arr[i], pair_ji * c_arr[j]],
            {"free_alpha": True},
        ),
        "free-alpha-symmetric": (
            [symmetric_ij, symmetric_ji * c_arr[i]],
            {
                "free_alpha": True,
                "transpose": True,
                "symmetric_params": ("kij",),
            },
        ),
        "free-alpha-nested": (
            [nested, nested + c_arr[l]],
            {"free_alpha": True},
        ),
        "no-sums": ([a_arr[i] + c_arr[j]], {"alpha": True}),
    }


def run_share(label):
    """Run ``_share_sums`` and return everything it produces."""
    expressions, options = share_cases()[label]
    namer = _Namer()
    arrays = {}
    inputs = list(expressions)
    residuals, definitions = cse._share_sums(
        expressions, namer, [i, j, k, l], arrays, sym_rank3=True, **options
    )
    assert expressions == inputs
    return residuals, definitions, namer.created, arrays


# --- _render_outer_fused ---------------------------------------------


def reduction(target, body, limits, **extra):
    """Return a reduction definition with no free indices."""
    return TempDef(target=target, free=[], limits=limits, body=body, **extra)


def component_map():
    """Return a component-fused map member."""
    return ComponentFusedMap(
        target=cmp0[i],
        free=[i],
        limits=[],
        body=sp.Integer(0),
        steps=[
            ("map", cmp0[i], a_arr[i] * 2),
            ("map", sp.Symbol("w"), cmp0[i] + b_arr[i]),
        ],
        members=[],
    )


def outer_cases():
    """Return ``label -> members`` for ``_render_outer_fused``."""
    over_i = (i, 1, NC)
    over_k = (k, 1, NC)
    over_l = (l, 1, NC)
    free_map = TempDef(
        target=u_arr[i], free=[i], limits=[], body=a_arr[i] * c_arr[i]
    )
    plain_a = reduction(sp.Symbol("s1"), a_arr[i] * b_arr[k], [over_i, over_k])
    plain_b = reduction(sp.Symbol("s2"), a_arr[i] + b_arr[k], [over_i, over_k])
    other = reduction(
        sp.Symbol("s3"), c_arr[i] * x_arr[i, l], [over_i, over_l]
    )
    single = reduction(sp.Symbol("s4"), a_arr[i] * c_arr[i], [over_i])
    factored = reduction(
        sp.Symbol("s5"),
        a_arr[i] * b_arr[k],
        [over_i, over_k],
        factored=True,
        acc=sp.Symbol("acc5"),
    )
    factored_b = reduction(
        sp.Symbol("s6"),
        c_arr[i] * x_arr[i, k],
        [over_i, over_k],
        factored=True,
        acc=sp.Symbol("acc6"),
    )
    return {
        "maps-only": [component_map(), free_map],
        "run-of-two": [plain_a, plain_b],
        "run-broken-by-limits": [plain_a, plain_b, other, single],
        "run-broken-by-map": [plain_a, component_map(), plain_b],
        "run-broken-by-free": [plain_a, free_map, plain_b],
        "factored-run": [factored, plain_a, factored_b, plain_b],
        "factored-alone": [factored],
        "mixed": [
            component_map(),
            single,
            plain_a,
            factored,
            free_map,
            other,
            factored_b,
        ],
        "empty": [],
    }


def run_outer(label):
    """Return the lines of ``_render_outer_fused`` and the members."""
    members = outer_cases()[label]
    snapshot = stable_text(members)
    nest = OuterFusedNest(
        target=sp.Symbol("t"),
        free=[],
        limits=[],
        body=sp.Integer(0),
        members=members,
        outer=i,
    )
    lines = render._render_outer_fused(nest)
    assert stable_text(members) == snapshot
    return lines


# --- _merge_dummy_images ---------------------------------------------


def merge_cases():
    """Return ``label -> (expression, options)`` for the merge."""
    triple = lam[k, l, m] * a_arr[k] * a_arr[l] * a_arr[m]
    cubic = sp.Sum(
        (b_arr[k] + b_arr[l] + b_arr[m]) * triple,
        (k, 1, nc),
        (l, 1, nc),
        (m, 1, nc),
    )
    pair = sp.Sum(
        (b_arr[k] + b_arr[l]) * kij[k, l] * a_arr[k] * a_arr[l],
        (k, 1, nc),
        (l, 1, nc),
    )
    return {
        "cubic-symmetric": (cubic, {"symmetric_params": ("lam",)}),
        "cubic-plain": (cubic, {}),
        "pair-symmetric": (pair, {"symmetric_params": ("kij",)}),
        "pair-plain": (pair, {}),
        "pair-symmetric-shared-base": (
            pair,
            {
                "symmetric_params": ("kij",),
                "shared_bases": frozenset({"a"}),
            },
        ),
        "pair-symmetric-other-shared-base": (
            pair,
            {
                "symmetric_params": ("kij",),
                "shared_bases": frozenset({"q"}),
            },
        ),
        "single-index": (
            sp.Sum(a_arr[k] * b_arr[k], (k, 1, nc)),
            {},
        ),
        "different-ranges": (
            sp.Sum(
                (b_arr[k] + b_arr[l]) * kij[k, l],
                (k, 1, nc),
                (l, 1, nc - 1),
            ),
            {"symmetric_params": ("kij",)},
        ),
        "four-indices": (
            sp.Sum(
                (b_arr[i] + b_arr[j] + b_arr[k] + b_arr[l]) * lam[i, j, k],
                (i, 1, nc),
                (j, 1, nc),
                (k, 1, nc),
                (l, 1, nc),
            ),
            {},
        ),
        "single-term": (
            sp.Sum(kij[k, l] * a_arr[k], (k, 1, nc), (l, 1, nc)),
            {"symmetric_params": ("kij",)},
        ),
        "nothing-to-merge": (
            sp.Sum(
                a_arr[k] * b_arr[l] + c_arr[k] * x_arr[k, l],
                (k, 1, nc),
                (l, 1, nc),
            ),
            {},
        ),
        "nested-in-sum": (
            sp.Sum(
                a_arr[i] * pair + c_arr[i] * pair,
                (i, 1, nc),
            ),
            {"symmetric_params": ("kij",)},
        ),
        "inside-expression": (
            c_arr[i] * pair + sp.sqrt(pair) + a_arr[j],
            {"symmetric_params": ("kij",)},
        ),
        "no-sum": (a_arr[i] * b_arr[j] + sp.Integer(3), {}),
        "atom": (sp.Symbol("q"), {}),
    }


def run_merge(label):
    """Run ``_merge_dummy_images`` and return its result."""
    expression, options = merge_cases()[label]
    return symmetry._merge_dummy_images(expression, sym_rank3=True, **options)


# --- end to end -------------------------------------------------------


def model_text(name):
    """Return the text of a test model or of an example."""
    if name == "pr":
        return EXAMPLES["Peng-Robinson (pressure)"]
    return (MODELS_DIR / name / f"{name}.model").read_text()


#: ``label`` -> (model, options)
END_TO_END = {
    "pr-standard": ("pr", {"preset": "standard"}),
    "pr-fastest": ("pr", {"preset": "fastest"}),
    "pr-no-sum-alpha": (
        "pr",
        {"preset": "fastest", "sum_alpha": False, "sum_free_alpha": False},
    ),
    "nrtl-fastest": ("nrtl", {"preset": "fastest"}),
    "nrtl-no-fuse-outer": ("nrtl", {"preset": "fastest", "fuse_outer": False}),
    "nrtl-no-transpose": (
        "nrtl",
        {"preset": "fastest", "sum_transpose": False},
    ),
    "uniquac-fastest": ("uniquac", {"preset": "fastest"}),
    "uniquac-no-free-alpha": (
        "uniquac",
        {"preset": "fastest", "sum_free_alpha": False},
    ),
    "uniquac-no-dummy-canon": (
        "uniquac",
        {"preset": "fastest", "dummy_canon": False},
    ),
    "cmr-fastest": ("cmr", {"preset": "fastest"}),
    "srk-fastest": ("srk", {"preset": "fastest"}),
}
HEAVY_END_TO_END = ("cmr-fastest", "uniquac-no-free-alpha", "srk-fastest")

#: kind:label -> digest of the pinned values
PINS = {
    "share:repeated": "d20d6252652432bb",
    "share:alpha": "01f7f49e3210513d",
    "share:no-alpha": "f7502a9eb8cf08ec",
    "share:nested": "11726e2b873f71a2",
    "share:free-alpha": "65e966cdd8ed9c22",
    "share:free-alpha-transpose": "4b42124d2b5ae546",
    "share:free-alpha-no-transpose": "6089bcb15d1917df",
    "share:free-alpha-symmetric": "056c02e71c631905",
    "share:free-alpha-nested": "03f9103e10210624",
    "share:no-sums": "89db27d671313053",
    "outer:maps-only": "9b01cfc9bdfdd42f",
    "outer:run-of-two": "b5c73ef908683389",
    "outer:run-broken-by-limits": "e86d18eafd294ba6",
    "outer:run-broken-by-map": "19970f87728dd88e",
    "outer:run-broken-by-free": "b64431e07d6fd47b",
    "outer:factored-run": "14dca321d385d6b1",
    "outer:factored-alone": "593ff1b715dde9ef",
    "outer:mixed": "04be7231eadf1f27",
    "outer:empty": "b9bd4170d4fcd300",
    "merge:cubic-symmetric": "63883fea664ee36a",
    "merge:cubic-plain": "3c341046c7260e03",
    "merge:pair-symmetric": "3f08961d881b07d4",
    "merge:pair-plain": "50fb9c40e7cf60a0",
    "merge:pair-symmetric-shared-base": "50fb9c40e7cf60a0",
    "merge:pair-symmetric-other-shared-base": "3f08961d881b07d4",
    "merge:single-index": "7cc79f40f0ead7b5",
    "merge:different-ranges": "ae536366f37a46a4",
    "merge:four-indices": "2810bcb5b4adab07",
    "merge:single-term": "2c9c58d47d96c7af",
    "merge:nothing-to-merge": "3dc2190abcfe46ee",
    "merge:nested-in-sum": "6645f772ebe012d9",
    "merge:inside-expression": "a9ebb7ff9b6fcb71",
    "merge:no-sum": "01c8492ee6714ec5",
    "merge:atom": "fdb082d4270ac766",
    "e2e:pr-standard": "dabaa4e42240b70e",
    "e2e:pr-fastest": "25a8b3fbfb4552df",
    "e2e:pr-no-sum-alpha": "bc698336f1830086",
    "e2e:nrtl-fastest": "b2aa7ffdfb3610c0",
    "e2e:nrtl-no-fuse-outer": "31b4cb7a14c33e6b",
    "e2e:nrtl-no-transpose": "6e711c8f6ba8f19d",
    "e2e:uniquac-fastest": "6b6fb31dd9328db5",
    "e2e:uniquac-no-free-alpha": "2bb756bd52e568f1",
    "e2e:uniquac-no-dummy-canon": "6b6fb31dd9328db5",
    "e2e:cmr-fastest": "ac3a42baf2b39c72",
    "e2e:srk-fastest": "9659190a766c0a09",
}


def end_to_end_params():
    """Parametrise the end-to-end cases."""
    return [
        pytest.param(
            label,
            marks=[pytest.mark.slow] if label in HEAVY_END_TO_END else [],
            id=label,
        )
        for label in END_TO_END
    ]


@pytest.mark.parametrize("label", share_cases())
def test_share_sums_is_pinned(label):
    assert digest(run_share(label)) == PINS[f"share:{label}"]


def test_share_sums_numbers_temporaries_in_order():
    residuals, definitions, created, arrays = run_share("repeated")
    assert [str(symbol) for symbol in created] == [
        f"t{pos}" for pos in range(1, len(created) + 1)
    ]
    assert len(definitions) == len(created)
    assert not any(expr.has(sp.Sum) for expr in residuals)


def test_share_sums_alpha_reuses_the_renamed_sum():
    plain = run_share("no-alpha")
    alpha = run_share("alpha")
    assert len(plain[1]) == 2 and len(alpha[1]) == 1


@pytest.mark.parametrize("label", outer_cases())
def test_render_outer_fused_is_pinned(label):
    assert digest(run_outer(label)) == PINS[f"outer:{label}"]


@pytest.mark.parametrize("label", merge_cases())
def test_merge_dummy_images_is_pinned(label):
    assert digest(run_merge(label)) == PINS[f"merge:{label}"]


def test_merge_dummy_images_merges_images_of_a_symmetric_sum():
    merged = run_merge("pair-symmetric")
    original = merge_cases()["pair-symmetric"][0]
    assert merged != original
    assert merged.func is sp.Sum


@pytest.mark.parametrize("label", end_to_end_params())
def test_generated_code_is_pinned(label):
    model, options = END_TO_END[label]
    code = compile_model(
        model_text(model), optional_derivatives=False, **options
    )
    text = hashlib.sha256(code.encode()).hexdigest()[:DIGEST_LENGTH]
    assert text == PINS[f"e2e:{label}"]
