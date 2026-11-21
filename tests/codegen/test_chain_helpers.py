"""Pins of four chain-rule helpers of ``chain.py``.

``abstract_component_functions``, ``_nest_inner_nodes``,
``_nested_node_derivatives`` and ``_propagate`` are pinned on small synthetic
inputs (the digest of every returned value, order included, plus the
registry or the counters they fill) and, end to end, through the hash of the
Fortran text of models where each of them runs.
"""

import hashlib
import re
from pathlib import Path

import pytest

import sympy as sp

from thermodiff import chain
from thermodiff.model import compile_model
from thermodiff.model.examples import EXAMPLES
from thermodiff.thermovars import T, V, k, l, m, n, nc

MODELS_DIR = Path(__file__).resolve().parents[1] / "data" / "models"
DUMMY_INDEX = re.compile(r"dummy_index=\d+")
DIGEST_LENGTH = 16
NC = nc
k_idx = k
l_idx = l
alpha = sp.IndexedBase("alpha")
kij = sp.IndexedBase("kij")
tau0 = sp.IndexedBase("tau0")
tau1 = sp.IndexedBase("tau1")
diam = sp.IndexedBase("diam")
ratio = sp.IndexedBase("ratio")


def digest(value):
    """Return the short SHA-256 of ``value`` without dummy counters."""
    text = DUMMY_INDEX.sub("dummy_index=0", stable_text(value))
    return hashlib.sha256(text.encode()).hexdigest()[:DIGEST_LENGTH]


def stable_text(value):
    """Return a deterministic text of nested SymPy containers."""
    if isinstance(value, dict):
        return "{%s}" % ", ".join(
            f"{stable_text(key)}: {stable_text(item)}"
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return "[%s]" % ", ".join(stable_text(item) for item in value)
    if isinstance(value, set):
        return "{%s}" % ", ".join(sorted(stable_text(it) for it in value))
    if isinstance(value, (sp.Basic, int, float)):
        return sp.srepr(sp.sympify(value))
    return repr(value)


def registry_summary(registry):
    """Return the ordered content of a registry of ``_Param`` objects."""
    return [
        (key, param.name, param.body, param.rank, param.symmetric)
        for key, param in registry.items()
    ]


# --- abstract_component_functions -------------------------------------


def double_sum(term):
    """Return ``sum_k sum_l term`` over the test indices."""
    return sp.Sum(term, (k_idx, 1, NC), (l_idx, 1, NC))


def cubic_mixing_body():
    """Return the body of a quadratic mixing rule with ``T`` dependence."""
    root_alpha = sp.sqrt(
        alpha[k_idx] * (1 + sp.sqrt(T)) * alpha[l_idx] * (1 + sp.sqrt(T))
    )
    return n[k_idx] * n[l_idx] * root_alpha * (1 - kij[k_idx, l_idx])


def table_body():
    """Return an NRTL-like body written with ``(l, k)`` subscripts."""
    tau = tau0[l_idx, k_idx] + tau1[l_idx, k_idx] / T
    return n[k_idx] * n[l_idx] * sp.exp(-tau * alpha[l_idx, k_idx])


def mixed_body():
    """Return a body with plain, volume-dependent and constant factors."""
    return (
        n[k_idx]
        * diam[k_idx]
        * (1 + T * ratio[k_idx])
        * sp.exp(T / V)
        * sp.log(diam[k_idx] * T)
        * sp.Piecewise((ratio[k_idx], T > 1), (0, True))
    )


ABSTRACT_CASES = {
    "mixing-default": (double_sum(cubic_mixing_body()), {}),
    "mixing-split-sqrt": (
        double_sum(cubic_mixing_body()),
        {"split_sqrt": True},
    ),
    "mixing-split-sqrt-symmetric": (
        double_sum(cubic_mixing_body()),
        {"split_sqrt": True, "symmetric_params": ("kij",)},
    ),
    "mixing-symmetric": (
        double_sum(cubic_mixing_body()),
        {"symmetric_params": ("kij",), "table_layout": True},
    ),
    "mixing-no-pairs": (double_sum(cubic_mixing_body()), {"pairs": False}),
    "table-default": (double_sum(table_body()), {}),
    "table-layout": (double_sum(table_body()), {"table_layout": True}),
    "table-layout-symmetric": (
        double_sum(table_body()),
        {
            "table_layout": True,
            "symmetric_params": ("tau0", "tau1", "alpha"),
        },
    ),
    "mixed-single-index": (
        sp.Sum(mixed_body(), (k_idx, 1, NC)),
        {},
    ),
    "pair-body-without-sum": (
        sp.sqrt(alpha[k_idx] * alpha[l_idx] * T) * (1 - kij[k_idx, l_idx]),
        {"split_sqrt": True},
    ),
    "custom-scheme": (
        double_sum(cubic_mixing_body()),
        {
            "scheme": chain.NameScheme(
                component_slot=sp.Idx("zc"),
                pair_slots=(sp.Idx("za"), sp.Idx("zb")),
                prefixes={chain.CMP_KIND: "vv", chain.MAT_KIND: "mm"},
            )
        },
    ),
}


def run_abstract(label):
    """Run one abstraction case; return result and registry summary."""
    expression, options = ABSTRACT_CASES[label]
    registry = {}
    result = chain.abstract_component_functions(
        expression, registry, **options
    )
    return result, registry_summary(registry)


# --- _nest_inner_nodes ------------------------------------------------


def inner_sum(index, power=3):
    """Return ``sum_index n_index d_index**power``."""
    return sp.Sum(n[index] * diam[index] ** power, (index, 1, NC))


def nesting_inputs():
    """Return the node lists of the nesting cases, keyed by label."""
    s_outer, s_known, s_other = (sp.Dummy(f"S{pos}") for pos in range(3))
    known = inner_sum(l)
    other = inner_sum(m, 2)
    outer = sp.Sum(n[k] * sp.log(1 - diam[k] * known), (k, 1, NC))
    two = sp.Sum(
        n[k] * (known + other**2) * ratio[k] / (1 + other), (k, 1, NC)
    )
    deep = sp.Sum(
        n[k] * sp.exp(-sp.Sum(n[m] * known * ratio[m], (m, 1, NC))),
        (k, 1, NC),
    )
    return {
        "existing-inner": [(s_outer, outer), (s_known, known)],
        "new-inner": [(s_outer, two)],
        "two-inner-one-known": [(s_outer, two), (s_known, known)],
        "nested-twice": [(s_outer, deep)],
        "no-inner": [(s_known, known), (s_other, other)],
    }


def run_nesting(label):
    """Nest one node list; return the nodes and the inner map."""
    nodes, inner_map = chain._nest_inner_nodes(nesting_inputs()[label])
    inner_map = [
        (symbol, sorted(kids, key=str)) for symbol, kids in inner_map.items()
    ]
    return nodes, inner_map


# --- _nested_node_derivatives -----------------------------------------

ALL_DERIVATIVES = [
    "dni",
    "dnidnj",
    "dt",
    "dt2",
    "dtdni",
    "dtdv",
    "dv",
    "dv2",
    "dvdni",
]
DERIVATIVE_SETS = {
    "all": ALL_DERIVATIVES,
    "first": ["dni", "dt", "dv"],
    "temperature": ["dt", "dt2"],
    "mixed": ["dtdni", "dvdni", "dni", "dt", "dv"],
    "none": [],
}


def nested_derivative_inputs():
    """Return ``(label, nodes, inner_map)`` of the derivative cases."""
    s_outer, s_known = sp.Dummy("S0"), sp.Dummy("S1")
    known = sp.Sum(n[l] * diam[l] ** 3 * T, (l, 1, NC))
    outer = sp.Sum(n[k] * sp.log(1 - diam[k] * known / V) * T, (k, 1, NC))
    two = sp.Sum(n[k] * (known + known**2) * ratio[k] / V, (k, 1, NC))
    return {
        "single-inner": [(s_outer, outer), (s_known, known)],
        "repeated-inner": [(s_outer, two)],
    }


NESTED_CASES = [
    (label, attrs)
    for label in ("single-inner", "repeated-inner")
    for attrs in DERIVATIVE_SETS
]


def run_nested_derivatives(label, attrs):
    """Return the derivatives of the nodes of one case."""
    nodes, inner_map = chain._nest_inner_nodes(
        nested_derivative_inputs()[label]
    )
    derivs = chain._nested_node_derivatives(
        nodes, inner_map, DERIVATIVE_SETS[attrs], [k, l, m]
    )
    return nodes, derivs


# --- _propagate -------------------------------------------------------

X, Y, Z = sp.symbols("x y z", positive=True)
W = sp.Symbol("w", positive=True)
PROPAGATE_CASES = {
    "smooth": (
        X * Y * sp.exp(X / Y) + sp.sqrt(X) * sp.log(Y) + Z,
        [X, Y, Z],
        [X, Y],
        [(X, X), (X, Y), (Y, Y)],
    ),
    "inactive-leaf": (
        X**2 * sp.sin(Y) + W * Z,
        [X, Y, Z, W],
        [Y],
        [(Y, Y)],
    ),
    "pairs-only": (
        (X + Y) ** 3 / (1 + Z * X),
        [X, Y, Z],
        [],
        [(X, Z), (Y, Y), (X, Y)],
    ),
    "not-explicit": (
        sp.Function("f")(X) * Y,
        [X, Y],
        [X],
        [(X, X)],
    ),
}


def run_propagate(label, mode):
    """Propagate one case; ``None`` when a partial is not explicit."""
    expression, leaves, first, pairs = PROPAGATE_CASES[label]
    return chain._propagate(expression, leaves, first, pairs, mode)


# --- end to end -------------------------------------------------------


def model_text(name):
    """Return the text of a test model or of an example."""
    if name == "pr":
        return EXAMPLES["Peng-Robinson (pressure)"]
    return (MODELS_DIR / name / f"{name}.model").read_text()


#: ``label`` -> (model, options, digest of the Fortran text)
END_TO_END = {
    "pr-standard": ("pr", {"preset": "standard"}),
    "pr-fastest": ("pr", {"preset": "fastest"}),
    "pr-no-split-sqrt": ("pr", {"preset": "fastest", "split_sqrt": False}),
    "nrtl-fastest": ("nrtl", {"preset": "fastest"}),
    "nrtl-no-table-layout": (
        "nrtl",
        {"preset": "fastest", "table_layout": False},
    ),
    "uniquac-fastest": ("uniquac", {"preset": "fastest"}),
    "uniquac-no-nested-nodes": (
        "uniquac",
        {"preset": "fastest", "nested_nodes": False},
    ),
}

#: kind:label -> digest of the pinned values
PINS = {
    "abs:mixing-default": "a3e7c7b77648e5e2",
    "abs:mixing-split-sqrt": "f35b8bc5366106e1",
    "abs:mixing-split-sqrt-symmetric": "f35b8bc5366106e1",
    "abs:mixing-symmetric": "71cb7fcd82177305",
    "abs:mixing-no-pairs": "83d43838e85a64f7",
    "abs:table-default": "6e44bcd62319e70f",
    "abs:table-layout": "310752047ad15532",
    "abs:table-layout-symmetric": "981e721bbd2123cf",
    "abs:mixed-single-index": "2056ee8b2e61b789",
    "abs:pair-body-without-sum": "dd102b63903fb183",
    "abs:custom-scheme": "5af93192c95057cb",
    "nest:existing-inner": "d79cf3770f57bac7",
    "nest:new-inner": "38368386892ddac2",
    "nest:two-inner-one-known": "88b0aa413cba848c",
    "nest:nested-twice": "964466f44eac9a81",
    "nest:no-inner": "aeb4c576ff852506",
    "der:single-inner:all": "c6a51921acccccd7",
    "der:single-inner:first": "d2a1acb35118a6d4",
    "der:single-inner:temperature": "b398bc5a9dbe0e88",
    "der:single-inner:mixed": "d059b1dc3f08c743",
    "der:single-inner:none": "887fcea1417d6800",
    "der:repeated-inner:all": "2d13956a2e0d3487",
    "der:repeated-inner:first": "1eab764547911d62",
    "der:repeated-inner:temperature": "6c462819958879c3",
    "der:repeated-inner:mixed": "605278c97305771b",
    "der:repeated-inner:none": "1b9be1e5a00c5f64",
    "prop:smooth:cse": "7b9bb5ace32932f8",
    "prop:smooth:fine": "34bf9cc22a8f5de8",
    "prop:inactive-leaf:cse": "eed7f00dde57bc7b",
    "prop:inactive-leaf:fine": "eed7f00dde57bc7b",
    "prop:pairs-only:cse": "3a6ae116b7f91ca8",
    "prop:pairs-only:fine": "f4526d85239a6274",
    "prop:not-explicit:cse": "dc937b59892604f5",
    "prop:not-explicit:fine": "dc937b59892604f5",
    "e2e:pr-standard": "dabaa4e42240b70e",
    "e2e:pr-fastest": "25a8b3fbfb4552df",
    "e2e:pr-no-split-sqrt": "4dfbd3581b5e9557",
    "e2e:nrtl-fastest": "b2aa7ffdfb3610c0",
    "e2e:nrtl-no-table-layout": "5f446c21e5122714",
    "e2e:uniquac-fastest": "6b6fb31dd9328db5",
    "e2e:uniquac-no-nested-nodes": "ef9ea3fa7d6b5b5b",
}
HEAVY_END_TO_END = ("uniquac-no-nested-nodes", "nrtl-no-table-layout")


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


@pytest.mark.parametrize("label", ABSTRACT_CASES)
def test_abstract_component_functions_is_pinned(label):
    assert digest(run_abstract(label)) == PINS[f"abs:{label}"]


def test_abstract_component_functions_registers_in_order():
    result, summary = run_abstract("mixing-split-sqrt")
    names = [name for _, name, _, _, _ in summary]
    assert names == [f"cmp{pos}" for pos in range(len(names))]
    assert not result.has(sp.sqrt(alpha[k] * alpha[l]))


@pytest.mark.parametrize("label", list(nesting_inputs()))
def test_nest_inner_nodes_is_pinned(label):
    assert digest(run_nesting(label)) == PINS[f"nest:{label}"]


def test_nest_inner_nodes_numbers_new_symbols_after_the_nodes():
    nodes, inner_map = run_nesting("new-inner")
    assert [str(symbol) for symbol, _ in nodes] == ["_S0", "_S1", "_S2"]
    assert dict(inner_map)[nodes[0][0]] == [nodes[1][0], nodes[2][0]]


@pytest.mark.parametrize("label,attrs", NESTED_CASES)
def test_nested_node_derivatives_are_pinned(label, attrs):
    result = run_nested_derivatives(label, attrs)
    assert digest(result) == PINS[f"der:{label}:{attrs}"]


@pytest.mark.parametrize("label", PROPAGATE_CASES)
@pytest.mark.parametrize("mode", ("cse", "fine"))
def test_propagate_is_pinned(label, mode):
    assert digest(run_propagate(label, mode)) == PINS[f"prop:{label}:{mode}"]


def test_propagate_gives_up_on_unevaluated_derivatives():
    assert run_propagate("not-explicit", "cse") is None


@pytest.mark.parametrize("label", end_to_end_params())
def test_generated_code_is_pinned(label):
    model, options = END_TO_END[label]
    code = compile_model(
        model_text(model), optional_derivatives=False, **options
    )
    text = hashlib.sha256(code.encode()).hexdigest()[:DIGEST_LENGTH]
    assert text == PINS[f"e2e:{label}"]
