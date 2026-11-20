"""Pin the ``sym_rank3`` canonicalisation of rank-3 symmetric parameters."""

import hashlib

import pytest
import sympy as sp

from thermodiff.codegen import pipeline
from thermodiff.fortran import _canonical_symmetric
from thermodiff.model import compile_model

RANK3_MODEL = """model r3
variables
    n nc
    T
parameters
    c nc nc nc symmetric
function
    T*sum_i sum_j sum_k n_i*n_j*n_k*c_ijk
"""

#: sha256 of the generated module, keyed by the sym_rank3 option
EXPECTED_DIGESTS = {
    False: "8525aa0ca489d4838dd06e1b3ddaf297d33ee2a807a5a965ed4b61d6197550d3",
    True: "82ef007513e777f3366ef009579a307a0755af31efda1ad7fb80a23bf4e33bcf",
}

#: pair function with a rank-3 symmetric parameter: the chain-rule stage
#: canonicalises it before the main pipeline reads the sym_rank3 option
CHAIN_MODEL = """model r3c
variables
    n nc
    T
parameters
    c nc nc nc symmetric
    a nc
function
    T*sum_i sum_j n_i*n_j*exp(-c_jji*T)*a_i
"""

#: sha256 of the chain-rule module generated with sym_rank3=False
EXPECTED_CHAIN_DIGEST = (
    "67c759511162e335373563758db258a17b30f7dc921a38caaeeb6bd2c6c918e6"
)


def _rank3_occurrence() -> sp.Expr:
    """Return ``c(k, i, j)`` with unsorted subscripts."""
    base = sp.IndexedBase("c")
    i, j, k = sp.symbols("i j k", cls=sp.Idx)
    return base[k, i, j]


def _canonical(sym_rank3: bool) -> sp.Expr:
    """Canonicalise the unsorted rank-3 occurrence."""
    return _canonical_symmetric(
        _rank3_occurrence(), ["c"], sym_rank3=sym_rank3
    )


def _generate(sym_rank3: bool) -> str:
    """Generate the rank-3 model with ``param_symmetry`` enabled."""
    return compile_model(
        RANK3_MODEL,
        derivatives="dn,dndn",
        preset="minimal",
        param_symmetry=True,
        sym_rank3=sym_rank3,
    )


def _digest(code: str) -> str:
    """Return the sha256 hex digest of generated code."""
    return hashlib.sha256(code.encode()).hexdigest()


def _generate_chain(sym_rank3: bool) -> str:
    """Generate the chain-rule model, whose chain stage runs first."""
    return compile_model(
        CHAIN_MODEL,
        derivatives="dn,dT",
        preset="minimal",
        param_symmetry=True,
        chain_rule=True,
        sym_rank3=sym_rank3,
    )


def _raise_midway(*args, **kwargs):
    """Stand-in for a later pipeline stage that fails."""
    raise RuntimeError("stage failed")


def test_rank3_sorted_when_flag_on():
    occurrence = _rank3_occurrence()
    expected = sp.Indexed(
        occurrence.base,
        *sorted(occurrence.indices, key=sp.default_sort_key),
    )
    assert _canonical(True) == expected
    assert _canonical(True) != occurrence


def test_rank3_untouched_when_flag_off():
    assert _canonical(False) == _rank3_occurrence()


def test_rank2_sorted_regardless_of_flag():
    base = sp.IndexedBase("c")
    i, j = sp.symbols("i j", cls=sp.Idx)
    expected = sp.Indexed(base, *sorted((i, j), key=sp.default_sort_key))
    for value in (False, True):
        occurrence = base[j, i]
        assert (
            _canonical_symmetric(occurrence, ["c"], sym_rank3=value)
            == expected
        )


@pytest.mark.parametrize("sym_rank3", [False, True])
def test_end_to_end_output_is_pinned(sym_rank3):
    code = compile_model(
        RANK3_MODEL,
        derivatives="dn,dndn",
        preset="minimal",
        param_symmetry=True,
        sym_rank3=sym_rank3,
    )
    digest = hashlib.sha256(code.encode()).hexdigest()
    assert digest == EXPECTED_DIGESTS[sym_rank3]


def test_failed_sym_rank3_call_does_not_leak(monkeypatch):
    with monkeypatch.context() as patched:
        patched.setattr(pipeline, "_fold_symmetric_sums", _raise_midway)
        with pytest.raises(RuntimeError, match="stage failed"):
            _generate_chain(True)
    # callers that run before the next emission see no leftover state
    occurrence = _rank3_occurrence()
    assert _canonical_symmetric(occurrence, ["c"]) == occurrence
    assert _digest(_generate_chain(False)) == EXPECTED_CHAIN_DIGEST
    assert _digest(_generate(False)) == EXPECTED_DIGESTS[False]


def test_interleaved_calls_match_separate_calls():
    interleaved = [
        _digest(_generate(flag)) for flag in (True, False, False, True)
    ]
    separate = [EXPECTED_DIGESTS[flag] for flag in (True, False, False, True)]
    assert interleaved == separate
