"""Pin the resolved settings of every benchmark variant key and alias.

``tests/data/variant_settings.json`` was written from the tables as they
were built incrementally (before the declarative registry).  Every public
table and ``variant_settings`` must keep returning exactly that content,
including the key order of each settings dict.
"""

import json
from pathlib import Path

import pytest

from benchmarks import td_bench
from benchmarks.td_bench import hooks

SNAPSHOT_PATH = Path(__file__).parent / "data" / "variant_settings.json"
SNAPSHOT = json.loads(SNAPSHOT_PATH.read_text())

PUBLIC_TABLES = (
    "VARIANTS",
    "EXTENDED_VARIANTS",
    "STAGE_VARIANTS",
    "LEGACY_KEYS",
    "WORKLOADS",
)


def _ordered(table):
    """Return ``table`` as nested lists of items, keeping key order."""
    return [
        (key, _ordered(value) if isinstance(value, dict) else value)
        for key, value in table.items()
    ]


@pytest.mark.parametrize("name", PUBLIC_TABLES)
def test_public_table_matches_snapshot(name):
    """Each public table has the frozen contents and key order."""
    assert _ordered(getattr(hooks, name)) == _ordered(SNAPSHOT[name])
    assert getattr(td_bench, name) == SNAPSHOT[name]


def test_extra_keys_match_snapshot():
    """The optional codegen keys keep their content and order."""
    assert list(hooks._EXTRA_KEYS) == SNAPSHOT["EXTRA_KEYS"]


@pytest.mark.parametrize("key", list(SNAPSHOT["resolved"]))
def test_variant_settings_by_key(key):
    """``variant_settings`` resolves every key and alias as frozen."""
    resolved = td_bench.variant_settings(key)
    assert _ordered(resolved) == _ordered(SNAPSHOT["resolved"][key])


@pytest.mark.parametrize("method_id", list(SNAPSHOT["resolved_by_id"]))
def test_variant_settings_by_method_id(method_id):
    """``variant_settings`` resolves every method id as frozen."""
    resolved = td_bench.variant_settings(method_id)
    expected = SNAPSHOT["resolved_by_id"][method_id]
    assert _ordered(resolved) == _ordered(expected)


def test_aliases_cover_snapshot():
    """Every legacy alias target is a resolvable key."""
    for legacy in SNAPSHOT["LEGACY_KEYS"].values():
        assert legacy in SNAPSHOT["resolved"]
