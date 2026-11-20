"""Code generation is re-entrant across interleaved calls and threads.

Every case is generated alone first (the reference); the same cases are
then generated interleaved in one thread, after a call that failed midway,
and concurrently, and each source must equal its reference byte for byte.
"""

import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from thermodiff import fortran
from thermodiff.model.compile import compile_model

from test_chain_names import forced_names_model

ROOT = Path(__file__).resolve().parent.parent
PR_TEXT = (ROOT / "examples" / "pr.model").read_text(encoding="utf-8")
SRK_TEXT = (
    ROOT / "tests" / "data" / "models" / "srk" / "srk.model"
).read_text(encoding="utf-8")
WORKLOAD_DERIVATIVES = {"w0": "none", "w1": "dV", "w2": "dT", "w3": "dn"}
THREAD_SEED = 20240607
THREAD_WORKERS = 6
THREAD_REPEATS = 2


class Case(NamedTuple):
    """One generation: model text, workload, preset and option overrides."""

    label: str
    text: str
    workload: str
    preset: str
    overrides: tuple[tuple[str, Any], ...] = ()


def build_cases() -> tuple[Case, ...]:
    """Return the cases; the forced-names model is built on demand."""
    forced = forced_names_model()
    return (
        Case("pr-min-w0", PR_TEXT, "w0", "minimal"),
        Case(
            "pr-std-w1-r3", PR_TEXT, "w1", "standard", (("sym_rank3", True),)
        ),
        Case(
            "pr-std-w2-nor3",
            PR_TEXT,
            "w2",
            "standard",
            (("sym_rank3", False),),
        ),
        Case("pr-fast-w3", PR_TEXT, "w3", "fastest"),
        Case(
            "srk-std-w3-est",
            SRK_TEXT,
            "w3",
            "standard",
            (("scalar_chain", "estimate"), ("optional_derivatives", False)),
        ),
        Case("srk-std-w2", SRK_TEXT, "w2", "standard"),
        Case(
            "srk-fast-w1-nor3",
            SRK_TEXT,
            "w1",
            "fastest",
            (("sym_rank3", False),),
        ),
        Case("forced-std-w3", forced, "w3", "standard"),
        Case(
            "forced-std-w2-est",
            forced,
            "w2",
            "standard",
            (("scalar_chain", "estimate"), ("optional_derivatives", False)),
        ),
        Case(
            "forced-fast-w1-nor3",
            forced,
            "w1",
            "fastest",
            (("sym_rank3", False),),
        ),
    )


def generate(case: Case) -> str:
    """Generate one case and return the Fortran module."""
    return compile_model(
        case.text,
        derivatives=WORKLOAD_DERIVATIVES[case.workload],
        preset=case.preset,
        **dict(case.overrides),
    )


@pytest.fixture(scope="module")
def cases() -> tuple[Case, ...]:
    """The cases under test."""
    return build_cases()


@pytest.fixture(scope="module")
def references(cases) -> dict[str, str]:
    """Source of each case generated alone, in the listed order."""
    return {case.label: generate(case) for case in cases}


def assert_matches(case: Case, source: str, references) -> None:
    """Assert one source equals the reference of its case."""
    assert source == references[case.label], case.label


def test_reference_is_stable_across_fresh_runs(cases, references):
    """Generating a case twice alone gives the same source."""
    for case in cases[:3]:
        assert_matches(case, generate(case), references)


def test_reference_cases_differ_from_each_other(references):
    """Guard: the cases are distinct, so the comparisons discriminate."""
    assert len(set(references.values())) == len(references)


def interleaving_orders(cases) -> dict[str, list[Case]]:
    """Return named orders: forward, reverse, alternating option sets."""
    forward = list(cases)
    by_model = {}
    for case in forward:
        by_model.setdefault(case.text is PR_TEXT, []).append(case)
    first, second = by_model[True], by_model[False]
    alternating = [
        case for pair in zip(first + first, second + second) for case in pair
    ]
    return {
        "forward": forward,
        "reverse": forward[::-1],
        "alternating": alternating,
        "stride": forward[::2] + forward[1::2],
    }


@pytest.mark.parametrize(
    "order", ["forward", "reverse", "alternating", "stride"]
)
def test_interleaved_calls_match_reference(order, cases, references):
    """Any order of generations in one thread reproduces the references."""
    for case in interleaving_orders(cases)[order]:
        assert_matches(case, generate(case), references)


@pytest.mark.slow
def test_failed_generation_does_not_poison_later_ones(
    cases, references, monkeypatch
):
    """A generation raising in a late stage leaves no state behind."""
    original = fortran._emit_code
    calls = {"count": 0}

    def failing_emit(*args, **kwargs):
        calls["count"] += 1
        raise RuntimeError("injected failure")

    for victim in (cases[1], cases[7], cases[4]):
        with monkeypatch.context() as patch:
            patch.setattr(fortran, "_emit_code", failing_emit)
            with pytest.raises(RuntimeError, match="injected failure"):
                generate(victim)
        assert fortran._emit_code is original
        for case in cases:
            assert_matches(case, generate(case), references)
    assert calls["count"] >= 3


def threaded_schedule(cases) -> list[Case]:
    """Each case repeated, shuffled with a fixed seed."""
    schedule = list(cases) * THREAD_REPEATS
    random.Random(THREAD_SEED).shuffle(schedule)
    return schedule


@pytest.mark.slow
def test_threaded_generation_matches_reference(cases, references):
    """Concurrent generations all reproduce their references."""
    schedule = threaded_schedule(cases)
    with ThreadPoolExecutor(max_workers=THREAD_WORKERS) as pool:
        sources = list(pool.map(generate, schedule))
    for case, source in zip(schedule, sources):
        assert_matches(case, source, references)
