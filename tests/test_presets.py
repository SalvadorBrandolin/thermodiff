"""Tests of the code-generation presets of the entry points."""

import sys
import warnings

import model_harness as harness
import numpy as np
import pytest
from fortran_tools import GFORTRAN

from thermodiff.codegen.options import (
    CodegenOptions,
    Preset,
    resolve_codegen_options,
)
from thermodiff.fortran import to_fortran
from thermodiff.model import compile_model, compile_model_file, examples
from thermodiff.model.cli import _build_parser
from thermodiff.model.cli import main as cli_main
from thermodiff.model.emit import build_diff, emit_module
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections


class TestResolveCodegenOptions:
    """Behaviour of :func:`resolve_codegen_options`."""

    def test_default_is_standard(self):
        assert resolve_codegen_options() == CodegenOptions.from_preset(
            "standard"
        )

    @pytest.mark.parametrize("preset", list(Preset))
    def test_preset_by_member_or_value(self, preset):
        expected = CodegenOptions.from_preset(preset)
        assert resolve_codegen_options(preset) == expected
        assert resolve_codegen_options(preset.value) == expected

    def test_options_replace_the_preset(self):
        options = CodegenOptions(sum_alpha=True)
        assert resolve_codegen_options(options=options) == options

    def test_preset_and_options_conflict(self):
        with pytest.raises(ValueError):
            resolve_codegen_options("fastest", CodegenOptions())

    def test_overrides_apply_and_none_is_ignored(self):
        resolved = resolve_codegen_options(
            "minimal", share=False, sum_alpha=None, array_cse=False
        )
        assert not resolved.share
        assert not resolved.array_cse
        assert not resolved.sum_alpha

    def test_none_keeps_the_preset_value(self):
        resolved = resolve_codegen_options("standard", sum_alpha=None)
        assert resolved.sum_alpha

    def test_unknown_override(self):
        with pytest.raises(TypeError, match="bogus"):
            resolve_codegen_options(bogus=True)


# ---------------------------------------------------------------------------
# entry points: minimal output, CLI flags and numerical agreement
# ---------------------------------------------------------------------------

SMALL_MODEL = """\
model small
variables
    n nc
    T
parameters
    b nc
function
    T*sum_i sum_j n_i*n_j*b_i*b_j
"""

GIBBS_MODEL = """\
model gex
variables
    n nc
    T
parameters
    A nc nc
expressions
    ntot = sum_i n_i
function
    sum_i n_i*log(n_i/ntot) + T*(sum_i sum_j n_i*n_j*A_ij)/ntot
"""

GIBBS_DERIVATIVES = "dT,dn,dT2,dndn,dndT"

PENG_ROBINSON = examples.get_example("Peng-Robinson (pressure)")

# Input expressions of the Fortran driver, one dict per state.
PR_STATES = (
    {
        "n": "0.3d0+0.11d0*i",
        "Tc": "190d0+37d0*i",
        "b": "0.02d0+0.003d0*i",
        "V": "2d0*sum(x_n)",
    },
    {
        "n": "0.2d0+0.07d0*i",
        "Tc": "150d0+21d0*i",
        "b": "0.03d0+0.002d0*i",
        "T": "250d0",
        "V": "3d0*sum(x_n)",
    },
)
GIBBS_STATES = (
    {"n": "0.3d0+0.11d0*i"},
    {"n": "0.8d0+0.05d0*i", "T": "280d0", "A": "0.1d0*i-0.07d0*j"},
)

RTOL_BY_PRESET = {"standard": 1e-10, "fastest": 1e-9}
ATOL = 1e-14


class TestEntryPoints:
    """The entry points resolve their options through one resolver."""

    def test_default_is_standard(self):
        assert compile_model(SMALL_MODEL) == compile_model(
            SMALL_MODEL, preset="standard"
        )
        assert compile_model(SMALL_MODEL) != compile_model(
            SMALL_MODEL, preset="minimal"
        )

    def test_minimal_matches_to_fortran_with_neutral_defaults(self):
        ir = build_ir(read_sections(SMALL_MODEL, None))
        diff, derivatives = build_diff(ir)
        body = to_fortran(
            diff,
            derivatives=derivatives,
            name="f",
            subroutine_name="small_derivs",
            integer_args=ir.dimensions,
            module_symbols=ir.imported,
            optional_derivatives=False,
            array_cse=True,
            share=True,
            symmetric_params=getattr(ir, "symmetric", ()),
        )
        indented = "\n".join(
            "   " + line if line.strip() else "" for line in body.splitlines()
        )
        assert indented in emit_module(ir, preset="minimal")

    def test_file_and_text_agree(self, tmp_path):
        path = tmp_path / "small.model"
        path.write_text(SMALL_MODEL)
        assert compile_model_file(str(path), preset="fastest") == (
            compile_model(SMALL_MODEL, preset="fastest")
        )

    def test_overrides_and_options(self):
        by_override = compile_model(
            SMALL_MODEL, preset="minimal", sum_alpha=True
        )
        by_options = compile_model(
            SMALL_MODEL, options=CodegenOptions(sum_alpha=True)
        )
        assert by_override == by_options

    def test_preset_and_options_conflict(self):
        with pytest.raises(ValueError):
            compile_model(
                SMALL_MODEL, preset="fastest", options=CodegenOptions()
            )

    def test_unknown_override(self):
        with pytest.raises(TypeError):
            compile_model(SMALL_MODEL, bogus=True)


class TestCli:
    """``--preset`` and the flags that override it only when given."""

    @staticmethod
    def _run(tmp_path, capsys, *flags):
        path = tmp_path / "small.model"
        path.write_text(SMALL_MODEL)
        assert cli_main([str(path), *flags]) == 0
        return capsys.readouterr().out

    def test_default_is_standard(self, tmp_path, capsys):
        assert self._run(tmp_path, capsys) == compile_model(
            SMALL_MODEL, filename="small.model", preset="standard"
        )

    @pytest.mark.parametrize("preset", ["minimal", "fastest"])
    def test_preset_flag(self, tmp_path, capsys, preset):
        assert self._run(
            tmp_path, capsys, "--preset", preset
        ) == compile_model(SMALL_MODEL, preset=preset)

    def test_no_share_overrides_fastest(self, tmp_path, capsys):
        shared = self._run(tmp_path, capsys, "--preset", "fastest")
        emitted = self._run(
            tmp_path, capsys, "--preset", "fastest", "--no-share"
        )
        expected = compile_model(SMALL_MODEL, preset="fastest", share=False)
        assert emitted == expected
        assert emitted != shared

    @pytest.mark.parametrize(
        ("flags", "overrides"),
        [
            (["--optional-derivatives"], {"optional_derivatives": True}),
            (["--no-array-cse"], {"array_cse": False}),
            (["--cse-backend", "sympy"], {"cse_backend": "sympy"}),
        ],
    )
    def test_flags_match_the_api_on_top_of_standard(
        self, tmp_path, capsys, flags, overrides
    ):
        emitted = self._run(tmp_path, capsys, "--preset", "standard", *flags)
        assert emitted == compile_model(
            SMALL_MODEL, preset="standard", **overrides
        )

    def test_unset_flags_keep_the_preset(self):
        args = _build_parser().parse_args(["m.model"])
        overrides = {
            name: value
            for name, value in vars(args).items()
            if name not in ("model", "output", "derivatives", "preset")
        }
        assert set(overrides.values()) == {None}


@pytest.mark.slow
@pytest.mark.skipif(GFORTRAN is None, reason="gfortran not available")
@pytest.mark.parametrize("preset", ["standard", "fastest"])
@pytest.mark.parametrize(
    ("text", "derivatives", "states"),
    [
        (PENG_ROBINSON, harness.W6, PR_STATES),
        (GIBBS_MODEL, GIBBS_DERIVATIVES, GIBBS_STATES),
    ],
    ids=["helmholtz", "gibbs"],
)
def test_preset_agrees_with_minimal_numerically(
    preset, text, derivatives, states
):
    """Every output of ``preset`` matches ``minimal`` (gfortran, -O0)."""
    for inits in states:
        reference, _ = harness.run(
            text,
            {"preset": "minimal"},
            inits,
            derivs=derivatives,
            opt=False,
        )
        values, _ = harness.run(
            text, {"preset": preset}, inits, derivs=derivatives, opt=False
        )
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(
            values, reference, rtol=RTOL_BY_PRESET[preset], atol=ATOL
        )


class TestExplicitOptionWarnings:
    """Only options the caller set explicitly are reported as ignored."""

    def test_preset_options_are_disabled_silently(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            resolved = resolve_codegen_options("standard", share=False)
        assert not resolved.sum_alpha
        assert not resolved.sum_free_alpha

    def test_explicit_option_is_reported(self):
        with pytest.warns(UserWarning, match="sum_alpha") as record:
            resolved = resolve_codegen_options(
                "minimal", sum_alpha=True, share=False
            )
        assert len(record) == 1
        assert not resolved.sum_alpha
        assert record[0].filename == __file__

    def test_explicit_options_object_is_reported(self):
        with pytest.warns(UserWarning, match="sum_alpha"):
            resolve_codegen_options(
                options=CodegenOptions(sum_alpha=True, share=False)
            )

    def test_fastest_with_optional_derivatives_is_silent(self):
        gated = (
            "lowrank_hess",
            "drop_pair_tables",
            "share_elementwise",
            "split_affine",
            "fuse_loops",
            "sib_fuse",
            "diag_table",
            "comp_fuse",
            "fuse_outer",
            "split_acc",
            "scalar_chain",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            resolved = resolve_codegen_options(
                "fastest", optional_derivatives=True
            )
        defaults = CodegenOptions()
        for name in gated:
            assert getattr(resolved, name) == getattr(defaults, name), name

    def test_entry_point_warning_points_at_the_caller(self):
        with pytest.warns(UserWarning, match="sum_alpha") as record:
            compile_model(
                SMALL_MODEL, preset="minimal", sum_alpha=True, share=False
            )
        assert record[0].filename == __file__

    def test_to_fortran_stays_silent_on_resolved_options(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compile_model(SMALL_MODEL, share=False)


class TestPresetOptionsConflict:
    """``preset`` and ``options`` are mutually exclusive, even "standard"."""

    @pytest.mark.parametrize("preset", ["minimal", "standard", "fastest"])
    def test_resolver_rejects_both(self, preset):
        with pytest.raises(ValueError, match="not both"):
            resolve_codegen_options(preset, CodegenOptions())

    def test_entry_point_rejects_explicit_standard(self):
        with pytest.raises(ValueError, match="not both"):
            compile_model(
                SMALL_MODEL, preset="standard", options=CodegenOptions()
            )

    def test_none_preset_means_standard(self):
        assert resolve_codegen_options(None) == CodegenOptions.from_preset(
            "standard"
        )


def test_caller_stacklevel_skips_frames_by_module_name(tmp_path):
    """The warning points at the caller even through ``compile_model_file``."""
    from thermodiff.codegen import options as options_module

    assert options_module._in_entry_point_module(sys._getframe()) is False
    path = tmp_path / "small.model"
    path.write_text(SMALL_MODEL)
    with pytest.warns(UserWarning, match="sum_alpha") as record:
        compile_model_file(
            str(path),
            preset="minimal",
            sum_alpha=True,
            share=False,
        )
    assert record[0].filename == __file__
