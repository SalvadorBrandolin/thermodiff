"""Streamlit independent logic behind the model GUI.

Everything in this module is plain Python: the GUI only wires widgets to
these helpers.  Keeping it free of any Streamlit import means the helpers
can be unit tested without installing the optional ``gui`` extra.
"""

from thermodiff.model.emit import build_diff, emit_module
from thermodiff.model.errors import ModelError
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections
from thermodiff.model.preview import latex_preview

__all__ = [
    "assemble_model",
    "build_from_parts",
    "build_result",
    "parse_parts",
    "annotated_source",
    "error_context",
]


def _declaration(row):
    """Render one variable/parameter row as a declaration line."""
    name = str(row.get("name") or "").strip()
    dims = str(row.get("dims") or "").strip().replace(",", " ")
    if not name:
        return ""
    return name if not dims else f"{name} {' '.join(dims.split())}"


def assemble_model(
    model_name, uses, variables, parameters, expressions, function
):
    """Build the text of a ``.model`` file from structured parts.

    Parameters
    ----------
    model_name : str
        Name of the model.  Empty falls back to ``model``.
    uses : list of str
        ``use`` statements, without the leading keyword.
    variables : list of dict
        Rows with ``name`` and ``dims`` keys.
    parameters : list of dict
        Rows with ``name`` and ``dims`` keys.
    expressions : list of dict
        Rows with ``lhs`` and ``rhs`` keys.
    function : str
        Body of the ``function`` section.

    Returns
    -------
    str
        The assembled ``.model`` text.
    """
    lines = []
    name = str(model_name or "").strip() or "model"
    lines.append("model " + name)
    lines.append("target fortran")

    for use in uses or []:
        use = str(use).strip()
        if not use:
            continue
        # Accept both "mod, only: x" and "use mod, only: x".
        if use.lower().startswith("use "):
            use = use[4:].strip()
        lines.append("use " + use)

    for header, rows in (("variables", variables), ("parameters", parameters)):
        declarations = [
            declaration
            for declaration in (_declaration(row) for row in rows or [])
            if declaration
        ]
        if declarations:
            lines.append("")
            lines.append(header)
            lines.extend("    " + declaration for declaration in declarations)

    definitions = []
    for row in expressions or []:
        lhs = str(row.get("lhs") or "").strip()
        rhs = str(row.get("rhs") or "").strip()
        if lhs and rhs:
            definitions.append(f"    {lhs} = {rhs}")
    if definitions:
        lines.append("")
        lines.append("expressions")
        lines.extend(definitions)

    lines.append("")
    lines.append("function")
    for line in str(function or "").splitlines():
        line = line.strip()
        if line:
            lines.append("    " + line)
    lines.append("")
    return "\n".join(lines)


def build_result(text, filename="<editor>", optional_derivatives=False):
    """Compile ``.model`` text and collect everything the GUI shows.

    This never raises: compilation problems are reported in the ``error``
    entry of the result.

    Parameters
    ----------
    text : str
        Contents of the ``.model`` file.
    filename : str, optional
        Name used in error messages.
    optional_derivatives : bool, optional
        Declare the outputs as Fortran ``optional`` arguments so the
        generated subroutine computes only the derivatives the caller
        passes.

    Returns
    -------
    dict
        With keys ``text``, ``code`` (Fortran or ``None``), ``latex``
        (preview or ``None``), ``warnings`` and ``error``.
    """
    try:
        raw = read_sections(text, filename)
        ir = build_ir(raw, filename)
        diff, derivatives = build_diff(ir)
        code = emit_module(
            ir,
            derivatives=derivatives,
            diff=diff,
            optional_derivatives=optional_derivatives,
        )
        preview = latex_preview(diff, derivatives)
        warnings = list(ir.warnings)
    except ModelError as error:
        return {
            "text": text,
            "code": None,
            "latex": None,
            "warnings": [],
            "error": {
                "message": error.message,
                "line": error.line,
                "formatted": str(error),
            },
        }
    return {
        "text": text,
        "code": code,
        "latex": preview,
        "warnings": warnings,
        "error": None,
    }


def build_from_parts(
    model_name,
    uses,
    variables,
    parameters,
    expressions,
    function,
    optional_derivatives=False,
):
    """Assemble structured parts and compile them in one step.

    Parameters
    ----------
    model_name, uses, variables, parameters, expressions, function
        See :func:`assemble_model`.
    optional_derivatives : bool, optional
        See :func:`build_result`.

    Returns
    -------
    dict
        The :func:`build_result` dictionary.
    """
    text = assemble_model(
        model_name, uses, variables, parameters, expressions, function
    )
    return build_result(text, optional_derivatives=optional_derivatives)


def parse_parts(text):
    """Split ``.model`` text back into GUI friendly rows.

    This is the inverse of :func:`assemble_model`, used to load the bundled
    examples into the editor widgets.

    Parameters
    ----------
    text : str
        Contents of a ``.model`` file.

    Returns
    -------
    dict
        With keys ``model_name``, ``uses``, ``variables``, ``parameters``,
        ``expressions`` and ``function``.  The keys match the parameters
        of :func:`assemble_model`, so ``assemble_model(**parts)`` rebuilds
        the text.

    Raises
    ------
    ModelError
        If the text is malformed and cannot be split into sections.
    """
    raw = read_sections(text, "<example>")
    return {
        "model_name": raw.model or "",
        "uses": [use for use, _line in raw.uses],
        "variables": [
            {"name": decl.name, "dims": " ".join(decl.dims)}
            for decl in raw.variables
        ],
        "parameters": [
            {"name": decl.name, "dims": " ".join(decl.dims)}
            for decl in raw.parameters
        ],
        "expressions": [
            {"lhs": definition.lhs, "rhs": definition.rhs}
            for definition in raw.expressions
        ],
        "function": raw.function.rhs if raw.function else "",
    }


def annotated_source(text, line=None):
    """Return ``text`` with line numbers, marking ``line`` with an arrow.

    Parameters
    ----------
    text : str
        Source text.
    line : int, optional
        Line number to highlight.

    Returns
    -------
    str
        A single string ready to be shown in a code block.
    """
    rendered = []
    for number, content in enumerate(str(text).splitlines(), start=1):
        marker = ">" if number == line else " "
        rendered.append(f"{marker} {number:>4} | {content}")
    return "\n".join(rendered)


def error_context(error, text):
    """Return the source line a compilation error points at.

    Parameters
    ----------
    error : dict
        The ``error`` entry of a :func:`build_result` dictionary.
    text : str
        Source text the error refers to.

    Returns
    -------
    str or None
        The offending line, without its line number, or ``None`` when the
        error has no location.
    """
    if not error or error.get("line") is None:
        return None
    lines = str(text).splitlines()
    line = error["line"]
    if 1 <= line <= len(lines):
        return lines[line - 1]
    return None
