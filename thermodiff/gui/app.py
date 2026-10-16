"""Streamlit front-end for the ``.model`` language.

Run it with::

    thermodiff-gui

or, equivalently::

    streamlit run thermodiff/gui/app.py

The application is intentionally built out of structured widgets (tables
with add/delete controls, pickers and buttons) instead of a bare text
editor: the user describes the model, the app shows the generated Fortran
and a typeset preview of every derivative.
"""

import pandas as pd

import streamlit as st

from thermodiff.gui import logic
from thermodiff.model.examples import (
    DEFAULT_EXAMPLE,
    example_names,
    get_example,
)

st.set_page_config(
    page_title="thermodiff model studio",
    page_icon="🧪",
    layout="wide",
)

# Compiling a model is the expensive step (symbolic differentiation).  The
# result only depends on the assembled text, so caching makes live editing
# cheap once a given text has been compiled.
_compile = st.cache_data(show_spinner=False)(logic.build_result)

_HELP = r"""
**Variables** are the things the function depends on. Use the names
`n`, `T`, `V` and `P`; `n` is the composition vector.

**Index notation**: write `x_i`, `k_ij`, `a_ijk`... The number of
subscripts must match the declared rank.

**Sums**: `sum_i expr` runs `i` over `1..nc`. Bounds can be given with
`sum_i[1:nc]`, `sum_i[2:]` or `sum_i[:5]`.

**Operators**: `+ - * /`, powers with `^` or `**`, and implicit
multiplication (`n_i T` is `n_i*T`).

**Functions**: `sqrt exp log ln sin cos tan abs min max erf erfc`.

**Expressions** are intermediate values (`amix = ...`); the final
result is the `function`.

**Parameters** are declared like variables but are inputs of the
generated subroutine.

Both a function and *all* its first and second derivatives (with respect
to the declared variables) are generated automatically.
"""


def _records(frame):
    """Turn an edited DataFrame into a list of clean string rows."""
    rows = []
    for record in frame.to_dict("records"):
        rows.append(
            {
                key: "" if pd.isna(value) else str(value)
                for key, value in record.items()
            }
        )
    return rows


def _editor(data, **kwargs):
    """Render ``st.data_editor`` at full width across Streamlit versions.

    Newer Streamlit versions deprecated ``use_container_width`` in favour
    of ``width``.  Try the modern spelling first and fall back so the GUI
    keeps working on older releases.
    """
    try:
        return st.data_editor(data, width="stretch", **kwargs)
    except TypeError:
        return st.data_editor(data, use_container_width=True, **kwargs)


def _declaration_editor(rows, key, help_text):
    """Render an editable table of ``name``/``dims`` declarations."""
    frame = pd.DataFrame(rows or [], columns=["name", "dims"])
    edited = _editor(
        frame,
        key=key,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "name": st.column_config.TextColumn(
                "Name", help=help_text, width="small"
            ),
            "dims": st.column_config.TextColumn(
                "Dimensions",
                help="Space separated dimensions, e.g. 'nc' or 'nc nc'.",
                width="medium",
            ),
        },
    )
    return _records(edited)


def _expression_editor(rows, key):
    """Render an editable table of ``lhs = rhs`` definitions."""
    frame = pd.DataFrame(rows or [], columns=["lhs", "rhs"])
    edited = _editor(
        frame,
        key=key,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "lhs": st.column_config.TextColumn(
                "Name",
                help="Name being defined, e.g. 'amix' or 'a_i'.",
                width="small",
            ),
            "rhs": st.column_config.TextColumn(
                "Expression",
                help="Right hand side, e.g. 'sum_i n_i*b_i'.",
                width="large",
            ),
        },
    )
    return _records(edited)


def main():
    """Render the whole application."""
    if "parts" not in st.session_state:
        st.session_state.parts = logic.parse_parts(
            get_example(DEFAULT_EXAMPLE)
        )
        st.session_state.revision = 0

    revision = st.session_state.revision
    parts = st.session_state.parts

    st.title("🧪 thermodiff model studio")
    st.caption(
        "Describe a thermodynamic model in index notation and get a "
        "Fortran module with the function and all of its analytical "
        "derivatives."
    )

    # ------------------------------------------------------------------
    # Sidebar: examples and help
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Examples")
        selected = st.selectbox(
            "Load an example",
            example_names(),
            index=example_names().index(DEFAULT_EXAMPLE),
            label_visibility="collapsed",
        )
        cols = st.columns(2)
        if cols[0].button("Load"):
            st.session_state.parts = logic.parse_parts(get_example(selected))
            st.session_state.revision += 1
            st.rerun()
        if cols[1].button("Reset"):
            st.session_state.parts = logic.parse_parts(
                get_example(DEFAULT_EXAMPLE)
            )
            st.session_state.revision += 1
            st.rerun()
        st.caption(
            "Loading an example replaces everything currently in the "
            "editor."
        )

        st.divider()
        st.header("Model language")
        st.markdown(_HELP)

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------
    top_left, top_right = st.columns([1, 3])
    model_name = top_left.text_input(
        "Model name",
        value=parts["model_name"],
        key=f"model_name_{revision}",
        help="Used for the module and subroutine names.",
    )
    uses_text = top_right.text_area(
        "use statements (one per line)",
        value="\n".join(parts["uses"]),
        key=f"uses_{revision}",
        height=80,
        help="Fortran modules required by the generated code, e.g. "
        "'yaeos__constants, only: R'.",
    )

    with st.expander("Variables", expanded=True):
        variables = _declaration_editor(
            parts["variables"],
            key=f"variables_{revision}",
            help_text="n (with dimension nc), T, V, P...",
        )
    with st.expander("Parameters", expanded=True):
        parameters = _declaration_editor(
            parts["parameters"],
            key=f"parameters_{revision}",
            help_text="Model inputs, e.g. 'ac' with dimension nc.",
        )
    with st.expander("Expressions", expanded=True):
        expressions = _expression_editor(
            parts["expressions"], key=f"expressions_{revision}"
        )
    function = st.text_area(
        "Function",
        value=parts["function"],
        key=f"function_{revision}",
        height=90,
        help="The expression to differentiate.",
    )

    # Keep the last known rows so reloading the page (or an example) does
    # not lose them.
    parts["variables"] = variables
    parts["parameters"] = parameters
    parts["expressions"] = expressions

    uses = [line for line in uses_text.splitlines() if line.strip()]
    text = logic.assemble_model(
        model_name, uses, variables, parameters, expressions, function
    )

    with st.spinner("Differentiating the model..."):
        result = _compile(text)

    _render_status(result)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    tab_fortran, tab_latex, tab_source = st.tabs(
        ["Fortran", "LaTeX preview", "Model source"]
    )

    with tab_fortran:
        _render_fortran(result)

    with tab_latex:
        _render_latex(result)

    with tab_source:
        error_line = result["error"]["line"] if result["error"] else None
        st.code(logic.annotated_source(text, error_line), language="text")
        st.download_button(
            "Download .model",
            text,
            file_name=f"{model_name or 'model'}.model",
            mime="text/plain",
        )


def _render_status(result):
    """Show warnings and errors just above the output tabs."""
    if result["error"]:
        error = result["error"]
        st.error(f"**{error['message']}**")
        line = error["line"]
        if line is not None:
            st.caption(
                f"Line {line}: `{logic.error_context(error, result['text'])}`"
            )
    else:
        st.success("Model compiled successfully.")
        for warning in result["warnings"]:
            st.warning(warning)


def _render_fortran(result):
    """Show the generated Fortran module or the compilation error."""
    if result["code"] is None:
        st.info("Fix the errors above to generate the Fortran module.")
        return
    st.download_button(
        "Download .f90",
        result["code"],
        file_name="model.f90",
        mime="text/plain",
    )
    st.code(result["code"], language="fortran")


def _render_latex(result):
    """Show a typeset preview of the function and its derivatives."""
    preview = result["latex"]
    if preview is None:
        st.info("Fix the errors above to see the LaTeX preview.")
        return

    with st.expander("Function f", expanded=False):
        st.latex(preview["function"])

    entries = preview["derivatives"]
    if not entries:
        st.info("This model has no non-trivial derivatives to show.")
        return

    st.subheader("Derivatives")
    show_all = st.checkbox("Show every derivative at once")
    if show_all:
        for entry in entries:
            st.latex(f"{entry['title']} = {entry['latex']}")
    else:
        labels = [f"{entry['name']}  —  {entry['title']}" for entry in entries]
        choice = st.selectbox("Derivative", labels)
        entry = entries[labels.index(choice)]
        st.latex(f"{entry['title']} = {entry['latex']}")


main()
