"""Entry points to compile ``.model`` files into Fortran modules."""

from thermodiff.model.emit import emit_module
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

__all__ = ["compile_model", "compile_model_file"]


def compile_model(text, filename=None, derivatives=None):
    """Compile the contents of a ``.model`` file into a Fortran module.

    Parameters
    ----------
    text : str
        Contents of the ``.model`` file.
    filename : str, optional
        File name used for error messages.
    derivatives : list of str, optional
        Derivative attributes to emit.  By default all first and second
        order derivatives among the declared variables are emitted.

    Returns
    -------
    str
        The complete Fortran module.
    """
    raw = read_sections(text, filename)
    ir = build_ir(raw, filename)
    return emit_module(ir, derivatives=derivatives)


def compile_model_file(path, derivatives=None):
    """Compile a ``.model`` file into a Fortran module.

    Parameters
    ----------
    path : str
        Path to the ``.model`` file.
    derivatives : list of str, optional
        Derivative attributes to emit.

    Returns
    -------
    str
        The complete Fortran module.
    """
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    return compile_model(text, filename=path, derivatives=derivatives)
