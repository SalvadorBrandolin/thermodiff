"""Unit tests of the ordered pass list of the emit pipeline."""

import ast
import inspect

import pytest

from thermodiff.codegen import CodegenOptions, ScalarChainMode, pipeline

MAX_COMPLEXITY = 10
BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.BoolOp,
    ast.IfExp,
    ast.comprehension,
)

EXPECTED_ORDER = [
    "resolve_outputs",
    "canonicalise_outputs",
    "pair_components",
    "scan_signature",
    "declare_signature",
    "array_cse",
    "mirror_array_temps",
    "reserved_names",
    "fuse_temps",
    "hoist_array_temps",
    "share_sum_temps",
    "hoist_sum_temps",
    "scalar_cse",
    "alias_scalar_chain",
    "rendered_outputs",
    "definitions_list",
    "lowrank_outputs",
    "hoist_loops",
    "drop_pair_tables",
    "share_elementwise",
    "split_affine",
    "mirror_sums",
    "column_layouts",
    "fuse_loop_nests",
    "symv_parts",
    "fuse_siblings",
    "diagonal_tables",
    "permute_subscripts",
    "component_maps",
    "outer_loops",
    "split_accumulators",
    "sparse_guards",
    "collect_stats",
    "render_bodies",
    "temp_declarations",
    "assemble_text",
]

# Passes that a preset never enables: they need an explicit option value.
OPT_IN = {"sparse_guards"}

ALWAYS = {
    "resolve_outputs",
    "canonicalise_outputs",
    "pair_components",
    "scan_signature",
    "declare_signature",
    "reserved_names",
    "share_sum_temps",
    "alias_scalar_chain",
    "rendered_outputs",
    "definitions_list",
    "collect_stats",
    "render_bodies",
    "temp_declarations",
    "assemble_text",
}


def enabled_names(opts):
    """Return the names of the passes enabled for ``opts``."""
    return {stage.name for stage in pipeline._PASSES if stage.enabled(opts)}


def preset(name, **overrides):
    """Return a preset with ``optional_derivatives`` off."""
    return CodegenOptions.from_preset(
        name, optional_derivatives=False, **overrides
    )


def test_pass_names_are_unique_and_in_the_documented_order():
    names = [stage.name for stage in pipeline._PASSES]
    assert len(set(names)) == len(names)
    assert names == EXPECTED_ORDER


def test_namer_stages_keep_their_relative_order():
    names = [stage.name for stage in pipeline._PASSES]
    chain = [
        "array_cse",
        "hoist_array_temps",
        "share_sum_temps",
        "hoist_sum_temps",
        "scalar_cse",
        "lowrank_outputs",
        "hoist_loops",
        "drop_pair_tables",
        "share_elementwise",
        "split_affine",
        "fuse_loop_nests",
        "symv_parts",
        "fuse_siblings",
        "component_maps",
        "split_accumulators",
        "render_bodies",
    ]
    positions = [names.index(name) for name in chain]
    assert positions == sorted(positions)
    assert names.index("scan_signature") < names.index("array_cse")


def test_every_pass_is_callable_with_two_arguments():
    for stage in pipeline._PASSES:
        assert len(inspect.signature(stage.run).parameters) == 2
        assert isinstance(stage.enabled(CodegenOptions()), bool)


def test_minimal_enables_only_the_ungated_passes_and_the_cse():
    enabled = enabled_names(preset("minimal"))
    assert enabled == ALWAYS | {"array_cse", "scalar_cse"}


def test_defaults_without_share_disable_the_scalar_cse():
    assert "scalar_cse" not in enabled_names(CodegenOptions(share=False))
    options = CodegenOptions(share=False, scalar_cse=True)
    assert "scalar_cse" in enabled_names(options)


def test_fastest_enables_every_pass():
    assert enabled_names(preset("fastest")) == set(EXPECTED_ORDER) - OPT_IN


def test_sparse_parameters_enable_their_pass():
    options = preset("fastest", sparse_parameters=("v",))
    assert enabled_names(options) == set(EXPECTED_ORDER)


def test_optional_derivatives_disable_the_guarded_passes():
    options = CodegenOptions.from_preset("fastest").replace(
        optional_derivatives=True
    )
    disabled = set(EXPECTED_ORDER) - enabled_names(options) - OPT_IN
    assert disabled == {
        "lowrank_outputs",
        "drop_pair_tables",
        "share_elementwise",
        "split_affine",
        "fuse_loop_nests",
        "fuse_siblings",
        "diagonal_tables",
        "component_maps",
        "outer_loops",
        "split_accumulators",
    }


def test_param_symmetry_gates_its_passes():
    base = CodegenOptions(sym_sum2=True, sym_matvec=True, sym_perm=True)
    gated = {"mirror_array_temps", "mirror_sums", "symv_parts"}
    gated |= {"permute_subscripts"}
    assert not gated & enabled_names(base)
    assert gated <= enabled_names(base.replace(param_symmetry=True))


@pytest.mark.parametrize(
    "mode", [ScalarChainMode.OFF, ScalarChainMode.ON, ScalarChainMode.COST]
)
def test_scalar_chain_mode_does_not_change_the_gates(mode):
    options = preset("standard", scalar_chain=mode)
    assert enabled_names(options) == enabled_names(preset("standard"))


def test_gate_factory_rejects_an_unknown_option():
    with pytest.raises(ValueError, match="unknown option"):
        pipeline._gate("not_an_option")


def complexity(function):
    """Return the McCabe-style complexity of ``function``."""
    tree = ast.parse(inspect.getsource(function).lstrip())
    return 1 + sum(isinstance(node, BRANCH_NODES) for node in ast.walk(tree))


def test_emit_code_is_a_plain_loop_over_the_passes():
    assert complexity(pipeline._emit_code) <= MAX_COMPLEXITY
