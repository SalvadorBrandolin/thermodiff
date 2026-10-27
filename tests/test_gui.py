"""Tests for the Streamlit front-end logic.

The widgets themselves live in :mod:`thermodiff.gui.app`; everything that
can be checked without a running Streamlit server is exercised here.  The
Streamlit only tests are guarded so the suite keeps passing when the
optional ``gui`` extra is not installed.
"""

import ast
import os
import sys

import pytest

from thermodiff.gui import logic
from thermodiff.model import examples

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(REPO_ROOT, "thermodiff", "gui", "app.py")

_FILE_BACKED = [
    ("Peng-Robinson (pressure)", "examples/pr.model"),
    ("Mole fraction", "examples/molefrac.model"),
]


def _parts(name):
    return logic.parse_parts(examples.get_example(name))


def _error_line(result):
    return logic.error_context(result["error"], result["text"])


def test_embedded_examples_match_example_files():
    """The bundled copies must not drift from the files in examples/."""
    for name, relative in _FILE_BACKED:
        with open(
            os.path.join(REPO_ROOT, relative), encoding="utf-8"
        ) as handle:
            assert handle.read() == examples.EXAMPLES[name]


def test_get_example_falls_back_to_default():
    assert (
        examples.get_example("does not exist")
        == examples.EXAMPLES[examples.DEFAULT_EXAMPLE]
    )


@pytest.mark.parametrize("name", examples.example_names())
def test_assemble_is_a_fixed_point_of_parse(name):
    canonical = logic.assemble_model(**_parts(name))
    assert logic.assemble_model(**logic.parse_parts(canonical)) == canonical


@pytest.mark.parametrize("name", examples.example_names())
def test_every_example_compiles(name):
    result = logic.build_from_parts(**_parts(name))
    assert result["error"] is None
    assert result["code"]
    assert result["latex"]["derivatives"]


def test_optional_derivatives_flag_reaches_fortran():
    parts = _parts(examples.DEFAULT_EXAMPLE)
    result = logic.build_from_parts(**parts, optional_derivatives=True)
    assert result["error"] is None
    assert "real(8), optional, intent(out) :: f" in result["code"]
    assert "present(" in result["code"]


def test_optional_derivatives_defaults_off():
    result = logic.build_from_parts(**_parts(examples.DEFAULT_EXAMPLE))
    assert "optional" not in result["code"]


def test_blank_rows_are_dropped():
    text = logic.assemble_model(
        "m",
        ["   ", "yaeos__constants, only: R"],
        [{"name": "", "dims": ""}, {"name": "n", "dims": "nc"}],
        [],
        [
            {"lhs": "a", "rhs": ""},
            {"lhs": "", "rhs": "b"},
            {"lhs": "c", "rhs": "3"},
        ],
        "sum_i n_i",
    )
    assert "yaeos__constants" in text
    assert "    n nc" in text
    assert "    a =" not in text and "= b" not in text
    assert "    c = 3" in text


def test_dimension_commas_are_normalized():
    text = logic.assemble_model(
        "m",
        [],
        [{"name": "k", "dims": "nc, nc"}],
        [],
        [],
        "sum_i sum_j k_ij",
    )
    assert "    k nc nc" in text


def test_use_statements_accept_the_keyword():
    text = logic.assemble_model(
        "m",
        ["use yaeos__constants, only: R", "containing_ns"],
        [{"name": "T", "dims": ""}],
        [],
        [],
        "T",
    )
    assert "use yaeos__constants, only: R" in text
    assert "use containing_ns" in text
    assert "use use" not in text


def test_undefined_name_reports_line_and_context():
    text = logic.assemble_model(
        "m", [], [{"name": "n", "dims": "nc"}], [], [], "n_i + q"
    )
    result = logic.build_result(text)
    assert result["code"] is None
    assert result["latex"] is None
    assert "undefined name 'q'" in result["error"]["message"]
    assert _error_line(result) == "    n_i + q"


def test_missing_function_section_is_reported():
    result = logic.build_result("model m\nvariables\n    n nc\n")
    assert result["error"] is not None
    assert "mandatory" in result["error"]["message"]
    assert result["error"]["line"] is None


def test_unused_parameter_warning_is_reported():
    text = logic.assemble_model(
        "m",
        [],
        [{"name": "T", "dims": ""}],
        [{"name": "k", "dims": ""}],
        [],
        "T",
    )
    result = logic.build_result(text)
    assert result["error"] is None
    assert any("'k'" in warning for warning in result["warnings"])


def test_latex_preview_entries():
    result = logic.build_result(
        examples.get_example("Peng-Robinson (pressure)")
    )
    preview = result["latex"]
    assert preview["function"]
    keys = {entry["key"] for entry in preview["derivatives"]}
    assert {"dT", "dV", "dni", "dT2", "dTn"} <= keys
    for entry in preview["derivatives"]:
        assert entry["latex"]
        assert entry["title"]
        assert entry["name"].startswith("df")


def test_annotated_source_marks_the_error_line():
    rendered = logic.annotated_source("a\nb\nc", line=2).splitlines()
    assert rendered[0].startswith(" ")
    assert rendered[1].startswith(">") and rendered[1].endswith("b")
    assert "3" in rendered[2]


def test_error_context_without_location():
    assert logic.error_context(None, "a") is None
    assert logic.error_context({"line": None}, "a") is None
    assert logic.error_context({"line": 99}, "a\nb") is None


def test_latex_preview_skips_unknown_attributes():
    from thermodiff.model.emit import build_diff
    from thermodiff.model.ir import build_ir
    from thermodiff.model.parser import read_sections
    from thermodiff.model.preview import latex_preview

    text = examples.get_example("Mole fraction")
    diff, derivatives = build_diff(build_ir(read_sections(text)))

    entries = latex_preview(diff, derivatives)["derivatives"]
    assert [entry["key"] for entry in entries] == ["dni", "dn2"]

    picked = latex_preview(diff, ["bogus", "dni"])["derivatives"]
    assert [entry["key"] for entry in picked] == ["dni"]


def test_app_module_parses():
    """Guard against syntax errors without importing Streamlit."""
    with open(APP_PATH, encoding="utf-8") as handle:
        ast.parse(handle.read(), APP_PATH)


def test_cli_reports_missing_streamlit(monkeypatch, capsys):
    for module in ("streamlit", "streamlit.web", "streamlit.web.cli"):
        monkeypatch.setitem(sys.modules, module, None)

    from thermodiff.gui import cli

    assert cli.main([]) == 1
    assert "thermodiff[gui]" in capsys.readouterr().err


def test_cli_forwards_arguments_to_streamlit(monkeypatch):
    """The launch path, exercised with a stub Streamlit."""
    import types

    calls = {}

    def fake_main():
        calls["argv"] = list(sys.argv)

    streamlit = types.ModuleType("streamlit")
    web = types.ModuleType("streamlit.web")
    web.cli = types.SimpleNamespace(main=fake_main)
    streamlit.web = web
    monkeypatch.setitem(sys.modules, "streamlit", streamlit)
    monkeypatch.setitem(sys.modules, "streamlit.web", web)
    monkeypatch.setitem(sys.modules, "streamlit.web.cli", web.cli)

    from thermodiff.gui import cli

    assert cli.main(["--server.port", "8502"]) == 0
    assert calls["argv"][:2] == ["streamlit", "run"]
    assert calls["argv"][-2:] == ["--server.port", "8502"]
    assert calls["argv"][2].endswith("app.py")


def test_streamlit_app_compiles_default_example():
    """End to end smoke test; skipped without the optional extra."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH, default_timeout=180).run()
    assert not app.exception
    assert not app.error
    assert app.success
    assert app.code  # the Fortran module is rendered
