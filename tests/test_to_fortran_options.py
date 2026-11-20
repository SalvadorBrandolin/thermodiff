"""Tests of the ``options`` parameter of :func:`to_fortran`."""

import inspect
import warnings
from pathlib import Path

import pytest

from benchmarks import td_bench as bench_hooks
from thermodiff.codegen import CodegenOptions
from thermodiff.fortran import to_fortran
from thermodiff.model.emit import build_diff
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "pr": ROOT / "examples" / "pr.model",
    "srk": ROOT / "tests" / "data" / "models" / "srk" / "srk.model",
}
CHEAP_VARIANTS = ("e0", "e8")
HEAVY_VARIANTS = ("p22", "p33")
WORKLOADS = ("w0", "w3")
NON_OPTION_KEYS = ("key", "id", "token", "label", "legacy")
P33_IGNORED_COUNT = 12


def build_case(model, workload, variant, optional_derivatives=False):
    """Return ``(diff, context, legacy)`` for one generation case."""
    text = MODELS[model].read_text(encoding="utf-8")
    ir = build_ir(read_sections(text, str(MODELS[model])), str(MODELS[model]))
    derivatives = bench_hooks.WORKLOADS[workload]
    diff, _ = build_diff(ir, derivatives=derivatives)
    settings = bench_hooks.variant_settings(variant)
    context = {
        "derivatives": derivatives,
        "name": "f",
        "subroutine_name": "routine",
        "integer_args": ir.dimensions,
        "module_symbols": ir.imported,
        "symmetric_params": getattr(ir, "symmetric", ()),
    }
    legacy = {
        key: value
        for key, value in settings.items()
        if key not in NON_OPTION_KEYS
    }
    legacy["optional_derivatives"] = optional_derivatives
    return diff, context, legacy


def cases():
    """Parametrisation: cheap cases plain, heavy ones marked ``slow``."""
    params = []
    for model in MODELS:
        for workload in WORKLOADS:
            for variant in CHEAP_VARIANTS + HEAVY_VARIANTS:
                marks = [pytest.mark.slow] if variant in HEAVY_VARIANTS else []
                params.append(
                    pytest.param(
                        model,
                        workload,
                        variant,
                        marks=marks,
                        id=f"{model}-{workload}-{variant}",
                    )
                )
    return params


@pytest.mark.parametrize("model, workload, variant", cases())
def test_legacy_and_options_agree(model, workload, variant):
    diff, context, legacy = build_case(model, workload, variant)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = to_fortran(diff, **context, **legacy)
        from_options = to_fortran(
            diff,
            **context,
            options=CodegenOptions.from_legacy(**legacy),
        )
        mixed = to_fortran(diff, **context, options=CodegenOptions(), **legacy)
    assert from_options == expected
    assert mixed == expected


def test_explicit_keyword_overrides_options():
    diff, context, legacy = build_case("pr", "w3", "e8")
    legacy["chain_rule"] = False
    standard = CodegenOptions.from_preset("standard")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = to_fortran(diff, **context, **legacy)
        actual = to_fortran(
            diff,
            **context,
            options=standard,
            optional_derivatives=False,
            chain_rule=False,
        )
    assert actual == expected


@pytest.mark.slow
def test_p33_with_optional_outputs_warns_once():
    diff, context, legacy = build_case(
        "pr", "w3", "p33", optional_derivatives=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ignored = CodegenOptions.from_legacy(**legacy).validate()[1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        to_fortran(diff, **context, **legacy)
    user_warnings = [w for w in caught if w.category is UserWarning]
    assert len(ignored) == P33_IGNORED_COUNT
    assert len(user_warnings) == 1
    message = str(user_warnings[0].message)
    assert all(name in message for name in ignored)


@pytest.mark.slow
def test_scalar_chain_reentry_does_not_revalidate():
    diff, context, legacy = build_case("pr", "w3", "p33")
    assert legacy["scalar_chain"] == "estimate"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        to_fortran(diff, **context, **legacy)
    assert [w for w in caught if w.category is UserWarning] == []


def test_unknown_keyword_raises_type_error():
    diff, context, _ = build_case("pr", "w0", "e0")
    with pytest.raises(TypeError):
        to_fortran(diff, **context, no_such_option=True)


def test_signature_exposes_options_and_hides_private_parameters():
    parameters = inspect.signature(to_fortran).parameters
    assert "options" in parameters
    assert "_structured" not in parameters
    assert "_structure_only" not in parameters


def test_parameters_after_subroutine_name_are_keyword_only():
    parameters = list(inspect.signature(to_fortran).parameters.values())
    names = [parameter.name for parameter in parameters]
    first_keyword = names.index("subroutine_name") + 1
    tail = parameters[first_keyword:]
    assert tail
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in tail)


def test_positional_option_raises_type_error():
    diff, _, _ = build_case("pr", "w0", "e0")
    with pytest.raises(TypeError):
        to_fortran(diff, None, None, None, True)


def test_warning_is_attributed_to_the_caller():
    diff, context, legacy = build_case(
        "pr", "w0", "e8", optional_derivatives=True
    )
    legacy.update(lowrank_hess=True)
    with warnings.catch_warnings(record=True) as direct:
        warnings.simplefilter("always")
        to_fortran(diff, **context, **legacy)
    with warnings.catch_warnings(record=True) as method:
        warnings.simplefilter("always")
        diff.to_fortran(**context, **legacy)
    for caught in (direct, method):
        assert len(caught) == 1
        assert caught[0].filename == __file__
