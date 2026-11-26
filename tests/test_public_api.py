"""Tests of the public namespace of the package."""

import thermodiff


def test_every_name_in_all_is_defined():
    """``from thermodiff import *`` must not fail on a missing name."""
    missing = [
        name for name in thermodiff.__all__ if not hasattr(thermodiff, name)
    ]
    assert missing == []


def test_star_import_works():
    """The star import resolves every exported name."""
    namespace = {}
    exec("from thermodiff import *", namespace)
    assert "DiffPlz" in namespace
