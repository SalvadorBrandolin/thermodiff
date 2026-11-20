"""Tests of the code-generation presets of the entry points."""

import pytest

from thermodiff.codegen.options import (
    CodegenOptions,
    Preset,
    resolve_codegen_options,
)


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
