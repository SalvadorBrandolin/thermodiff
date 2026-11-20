"""Entry points to compile ``.model`` files into Fortran modules."""

from thermodiff.model.emit import emit_module
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

__all__ = ["compile_model", "compile_model_file"]


def compile_model(
    text,
    filename=None,
    derivatives=None,
    *,
    preset="standard",
    options=None,
    **overrides,
):
    """Compile the contents of a ``.model`` file into a Fortran module.

    Parameters
    ----------
    text : str
        Contents of the ``.model`` file.
    filename : str, optional
        File name used for error messages.
    derivatives : str or list of str, optional
        Derivatives to emit, for example ``"dn,dndn"`` or ``["dT"]``.
        Friendly names (``dT``, ``dn``, ``dndT``, ...), the groups
        ``all``, ``none``, ``first`` and ``second`` and the
        :class:`~thermodiff.diffplz.DiffPlz` attribute names are
        accepted.  By default every derivative applicable to the declared
        variables is emitted.
    preset : {"minimal", "standard", "fastest"}, optional
        Named set of code-generation options.  The default ``"standard"``
        is the recommended set; ``"minimal"`` reproduces the plain
        generator output of earlier versions; ``"fastest"`` gives the
        fastest generated code, with slower generation and last-bit
        rounding differences.
    options : CodegenOptions, optional
        Explicit options used instead of a preset.
    **overrides
        Individual options replacing those of the preset, for example
        ``optional_derivatives=True`` (declare every output as a Fortran
        ``optional`` argument), ``array_cse=False`` or ``share=False``.
        See :func:`thermodiff.model.emit.emit_module`.

    Returns
    -------
    str
        The complete Fortran module.
    """
    raw = read_sections(text, filename)
    ir = build_ir(raw, filename)
    return emit_module(
        ir,
        derivatives=derivatives,
        preset=preset,
        options=options,
        **overrides,
    )


def compile_model_file(
    path,
    derivatives=None,
    *,
    preset="standard",
    options=None,
    **overrides,
):
    """Compile a ``.model`` file into a Fortran module.

    Parameters
    ----------
    path : str
        Path to the ``.model`` file.
    derivatives : str or list of str, optional
        Derivatives to emit.  See :func:`compile_model`.
    preset : {"minimal", "standard", "fastest"}, optional
        Named set of code-generation options (default ``"standard"``).
        See :func:`compile_model`.
    options : CodegenOptions, optional
        Explicit options used instead of a preset.
    **overrides
        Individual options replacing those of the preset.  See
        :func:`compile_model`.

    Returns
    -------
    str
        The complete Fortran module.
    """
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    return compile_model(
        text,
        filename=path,
        derivatives=derivatives,
        preset=preset,
        options=options,
        **overrides,
    )
