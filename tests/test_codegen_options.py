"""Tests of :mod:`thermodiff.codegen.options`."""

import dataclasses
import inspect
import warnings

from benchmarks import td_bench as bench_hooks

import pytest

from thermodiff.codegen import (
    CodegenOptions,
    CseBackend,
    LowrankHessMode,
    PRESET_SETTINGS,
    Preset,
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
    "options",
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


class TestSplitAccFalsy:
    """Every falsy ``split_acc`` is off, as in ``if split_acc and ...``."""

    @pytest.mark.parametrize("value", [False, None, 0, 0.0, ""])
    def test_falsy_is_off(self, value):
        options = CodegenOptions.from_legacy(split_acc=value)
        assert options.split_acc is None
        assert options.to_legacy()["split_acc"] is False
        assert options.validate() == (options, ())


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

    def test_odd_truthy_warning_points_at_caller(self):
        with pytest.warns(UserWarning) as caught:
            CodegenOptions.from_legacy(sep_sums="bogus")
        assert caught[0].filename == __file__

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


def validated(**settings):
    """Validate options, returning the result and the emitted warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = CodegenOptions(**settings).validate()
    messages = [w for w in caught if issubclass(w.category, UserWarning)]
    return result, messages


# (condition settings, ignored option set on top, non-ignoring settings)
RULE_CASES = [
    (
        {"optional_derivatives": True},
        {
            "lowrank_hess": True,
            "drop_pair_tables": True,
            "share_elementwise": True,
            "split_affine": True,
            "fuse_loops": True,
            "sib_fuse": True,
            "diag_table": True,
            "comp_fuse": True,
            "split_acc": 2,
            "scalar_chain": "cost",
        },
        {"optional_derivatives": False},
    ),
    (
        {"chain_rule": False},
        {
            "split_sqrt": True,
            "split_sums": True,
            "pair_cse": True,
            "pair_rates": True,
            "nested_nodes": True,
            "scalar_chain": "estimate",
            "table_layout": True,
        },
        {"chain_rule": True},
    ),
    (
        {"share": False},
        {
            "sum_alpha": True,
            "sum_free_alpha": True,
            "sum_transpose": True,
        },
        {"share": True},
    ),
    (
        {"share_sums": False},
        {"sum_alpha": True},
        {"share_sums": True, "share": False},
    ),
    ({"array_cse": False}, {"cse_expensive": True}, {"array_cse": True}),
    (
        {"fuse_temps": False},
        {"fuse_guard": True, "keep_libm_tables": True},
        {"fuse_temps": True},
    ),
    (
        {"nest_sums": False},
        {"nest_general": True, "nest_cost": True},
        {"nest_sums": True},
    ),
    (
        {"fuse_temps": False, "nest_sums": False, "param_symmetry": False},
        {"nest_shared": True},
        {"fuse_temps": True},
    ),
    (
        {"param_symmetry": False},
        {"sym_matvec": True, "sym_sum2": True, "sym_perm": True},
        {"param_symmetry": True},
    ),
    (
        {"fuse_outer": False},
        {"fuse_libm_guard": True},
        {"fuse_outer": True},
    ),
]


class TestValidateRules:
    """Each rule disables, and reports, exactly its ignored options."""

    @pytest.mark.parametrize("gate, ignored, enabled", RULE_CASES)
    def test_rule_fires(self, gate, ignored, enabled):
        base = {"chain_rule": True, "share": True}
        settings = {**base, **gate, **ignored}
        (options, names), caught = validated(**settings)
        defaults = CodegenOptions()
        for name in names:
            assert getattr(options, name) == getattr(defaults, name)
        assert set(names) == set(ignored)
        assert names
        assert len(caught) == 1
        assert all(name in str(caught[0].message) for name in names)

    @pytest.mark.parametrize("gate, ignored, enabled", RULE_CASES)
    def test_rule_does_not_fire(self, gate, ignored, enabled):
        settings = {
            "chain_rule": True,
            "share": True,
            "optional_derivatives": False,
            "array_cse": True,
            "fuse_temps": True,
            "nest_sums": True,
            "param_symmetry": True,
            "fuse_outer": True,
            "col_major": True,
            **enabled,
            **ignored,
        }
        # an ignored option whose gate is open must be kept
        (options, names), caught = validated(**settings)
        assert names == ()
        assert caught == []
        assert options == CodegenOptions(**settings)

    def test_each_listed_option_is_reported_in_field_order(self):
        (_, names), _ = validated(
            optional_derivatives=True, lowrank_hess=True, split_affine=True
        )
        assert names == ("lowrank_hess", "split_affine")

    def test_table_layout_kept_with_col_major(self):
        (options, names), _ = validated(table_layout=True, col_major=True)
        assert options.table_layout
        assert names == ()

    def test_fixed_point_cascade(self):
        # optional_derivatives disables fuse_outer, which in turn makes
        # fuse_libm_guard moot; the cascaded name is listed last.
        (options, names), caught = validated(
            optional_derivatives=True, fuse_outer=True, fuse_libm_guard=True
        )
        assert names == ("fuse_outer", "fuse_libm_guard")
        assert not options.fuse_libm_guard
        assert len(caught) == 1
        assert "fuse_libm_guard" in str(caught[0].message)

    def test_scalar_chain_on_is_not_reported(self):
        (options, names), caught = validated(
            optional_derivatives=True, scalar_chain=True
        )
        assert options.scalar_chain is ScalarChainMode.ON
        assert names == ()
        assert caught == []

    def test_sym_rank3_is_never_touched(self):
        (options, names), _ = validated(
            sym_rank3=True, param_symmetry=False, sym_sum2=True
        )
        assert options.sym_rank3
        assert "sym_rank3" not in names
        assert not options.sym_sum2

    def test_model_dependent_options_do_not_warn(self):
        (_, names), caught = validated(
            param_symmetry=True, hessian_symmetry=True, chain_rule=True
        )
        assert names == ()
        assert caught == []

    def test_clean_options_are_unchanged(self):
        (options, names), caught = validated()
        assert options == CodegenOptions()
        assert names == () and caught == []


class TestValidateSplitAcc:
    """Only ``split_acc=2`` is legal where ``to_fortran`` checks it."""

    @pytest.mark.parametrize("value", [True, 1, 3])
    def test_illegal_value_raises(self, value):
        with pytest.raises(ValueError, match="2 accumulators only"):
            CodegenOptions(split_acc=value).validate()

    def test_two_is_legal(self):
        options, names = CodegenOptions(split_acc=2).validate()
        assert options.split_acc == 2 and names == ()

    def test_illegal_value_is_ignored_when_optional(self):
        (options, names), caught = validated(
            optional_derivatives=True, split_acc=3
        )
        assert options.split_acc is None
        assert names == ("split_acc",)
        assert len(caught) == 1


class TestPresets:
    """Presets reproduce the bench settings they stand for."""

    def test_minimal_is_defaults(self):
        options = CodegenOptions.from_preset("minimal")
        assert options.to_legacy() == signature_defaults()

    @pytest.mark.parametrize(
        "preset, variant_key",
        [("standard", "e8"), ("fastest", "p33")],
    )
    def test_matches_bench_variant(self, preset, variant_key):
        variant = bench_hooks.variant_settings(variant_key)
        expected = {**signature_defaults(), **codegen_settings(variant)}
        options = CodegenOptions.from_preset(preset)
        assert options.to_legacy() == expected

    def test_preset_member_and_overrides(self):
        options = CodegenOptions.from_preset(
            Preset.STANDARD, chain_rule=False, array_pad=8
        )
        assert not options.chain_rule
        assert options.array_pad == 8
        assert options.nest_sums

    def test_unknown_preset(self):
        with pytest.raises(ValueError):
            CodegenOptions.from_preset("turbo")

    def test_table_is_immutable(self):
        with pytest.raises(TypeError):
            PRESET_SETTINGS[Preset.MINIMAL] = {}
        with pytest.raises(TypeError):
            PRESET_SETTINGS[Preset.STANDARD]["chain_rule"] = False

    def test_setting_counts(self):
        assert len(PRESET_SETTINGS[Preset.MINIMAL]) == 0
        assert len(PRESET_SETTINGS[Preset.STANDARD]) == 14
        assert len(PRESET_SETTINGS[Preset.FASTEST]) == 44

    def test_standard_without_optional_derivatives_is_clean(self):
        options = CodegenOptions.from_preset("standard")
        _, caught = validated(**options.to_legacy())
        assert caught == []

    def test_fastest_without_optional_derivatives_is_clean(self):
        options = CodegenOptions.from_preset("fastest")
        (normalised, names), caught = validated(**options.to_legacy())
        assert names == () and caught == []
        assert normalised == options

    def test_fastest_with_optional_derivatives_warns(self):
        options = CodegenOptions.from_preset(
            "fastest", optional_derivatives=True
        )
        (normalised, names), caught = validated(**options.to_legacy())
        assert names == (
            "fuse_loops",
            "comp_fuse",
            "fuse_outer",
            "diag_table",
            "sib_fuse",
            "lowrank_hess",
            "drop_pair_tables",
            "share_elementwise",
            "scalar_chain",
            "split_affine",
            "split_acc",
            "fuse_libm_guard",
        )
        assert len(caught) == 1
        assert normalised.sym_rank3
