"""Console entry point that launches the Streamlit application."""

import os
import sys

__all__ = ["main"]

#: Message shown when the optional Streamlit dependency is missing.
_MISSING = (
    "The graphical front-end needs Streamlit, which is an optional "
    "dependency.\nInstall it with:\n\n    pip install 'thermodiff[gui]'\n"
)


def main(argv=None):
    """Launch the Streamlit GUI.

    Parameters
    ----------
    argv : list of str, optional
        Extra arguments forwarded to ``streamlit run``.

    Returns
    -------
    int
        Process exit code.
    """
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print(_MISSING, file=sys.stderr)
        return 1

    app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    sys.argv = ["streamlit", "run", app]
    if argv is not None:
        sys.argv.extend(argv)

    stcli.main()
    return 0
