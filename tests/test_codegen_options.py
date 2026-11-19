"""Tests of :mod:`thermodiff.codegen.options`."""

import dataclasses
import inspect
import warnings

import pytest

from thermodiff import bench_hooks
from thermodiff.codegen import (
    CodegenOptions,
    CseBackend,
    LowrankHessMode,
    ScalarChainMode,
    SepSumsMode,
)
from thermodiff.fortran import to_fortran

METADATA_KEYS = {"id", "key", "label", "legacy", "token"}
CONTEXT_PARAMETERS = {
    "diff",
    "derivatives",
    "name",
    "subroutine_name",
    "integer_args",
    "module_symbols",
    "symmetric_params",
    "stats",
    "_structured",
    "_structure_only",
}


def signature_defaults():
    """Return the ``to_fortran`` defaults of every option field."""
    parameters = inspect.signature(to_fortran).parameters
    return {
        name: parameter.default
        for name, parameter in parameters.items()
        if name not in CONTEXT_PARAMETERS
    }


def codegen_settings(variant):
    """Return the ``to_fortran`` keys of a bench variant dict."""
    return {k: v for k, v in variant.items() if k not in METADATA_KEYS}


def all_variants():
    """Return ``(key, settings)`` of every bench variant."""
    tables = (
        bench_hooks.VARIANTS,
        bench_hooks.EXTENDED_VARIANTS,
        bench_hooks.STAGE_VARIANTS,
    )
    return [
        (key, codegen_settings(variant))
        for table in tables
        for key, variant in table.items()
    ]


class TestLegacyRoundTrip:
    """``from_legacy`` and ``to_legacy`` reproduce today's keywords."""

    @pytest.mark.parametrize(
        "key, codegen", all_variants(), ids=[k for k, _ in all_variants()]
    )
    def test_every_bench_variant(self, key, codegen):
        options = CodegenOptions.from_legacy(**codegen)
        expected = {**signature_defaults(), **codegen}
        assert options.to_legacy() == expected

    def test_defaults_match_signature(self):
        assert CodegenOptions().to_legacy() == signature_defaults()

    def test_field_names_match_signature(self):
        names = [f.name for f in dataclasses.fields(CodegenOptions)]
        assert names == list(signature_defaults())

    def test_enum_values(self):
        options = CodegenOptions.from_legacy(
            scalar_chain="estimate",
            sep_sums="cost",
            lowrank_hess="sym",
            cse_backend="sympy",
            split_acc=2,
        )
        assert options.scalar_chain is ScalarChainMode.ESTIMATE
        assert options.sep_sums is SepSumsMode.COST
        assert options.lowrank_hess is LowrankHessMode.SYM
        assert options.cse_backend is CseBackend.SYMPY
        assert options.split_acc == 2

    def test_split_acc_legacy_forms(self):
        assert CodegenOptions(split_acc=False).split_acc is None
        assert CodegenOptions(split_acc=None).to_legacy()["split_acc"] is (
            False
        )
        assert CodegenOptions(split_acc="2").split_acc == 2
        assert CodegenOptions(split_acc=True).split_acc == 1

    def test_replace_coerces(self):
        options = CodegenOptions().replace(sep_sums=True, array_pad=8)
        assert options.sep_sums is SepSumsMode.ALL
        assert options.array_pad == 8


class TestLegacyErrors:
    """Unknown keys and values fail or warn as today."""

    def test_unknown_key(self):
        with pytest.raises(TypeError):
            CodegenOptions.from_legacy(not_an_option=True)

    def test_context_argument_is_not_an_option(self):
        with pytest.raises(TypeError):
            CodegenOptions.from_legacy(derivatives="dT")

    def test_bad_cse_backend(self):
        with pytest.raises(ValueError, match="unknown cse_backend 'foo'"):
            CodegenOptions.from_legacy(cse_backend="foo", scalar_cse=False)

    def test_non_integer_split_acc(self):
        with pytest.raises(ValueError):
            CodegenOptions.from_legacy(split_acc="two")

    @pytest.mark.parametrize(
        "name, value, expected",
        [
            ("scalar_chain", "bogus", ScalarChainMode.ON),
            ("sep_sums", "bogus", SepSumsMode.ALL),
            ("lowrank_hess", "bogus", LowrankHessMode.ON),
        ],
    )
    def test_odd_truthy_value_warns(self, name, value, expected):
        with pytest.warns(UserWarning, match=name):
            options = CodegenOptions.from_legacy(**{name: value})
        assert getattr(options, name) is expected

    @pytest.mark.parametrize(
        "name", ["scalar_chain", "sep_sums", "lowrank_hess"]
    )
    def test_known_values_do_not_warn(self, name):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CodegenOptions.from_legacy(**{name: True})
            CodegenOptions.from_legacy(**{name: None})


class TestImmutability:
    """Options are frozen."""

    def test_field_assignment_raises(self):
        options = CodegenOptions()
        with pytest.raises(dataclasses.FrozenInstanceError):
            options.chain_rule = True
