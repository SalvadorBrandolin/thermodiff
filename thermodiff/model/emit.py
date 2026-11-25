"""Fortran emission for the ``.model`` front-end.

The heavy lifting is delegated to :func:`thermodiff.fortran.to_fortran`; this
module only takes care of the things that are specific to a whole model:

* renaming the user indices to friendly loop variables that cannot clash
  with the generated arguments,
* choosing which derivatives to emit,
* wrapping the generated subroutine in a module together with the ``use``
  statements requested in the ``.model`` file.
"""

import sympy as sp

from thermodiff.codegen.options import resolve_codegen_options
from thermodiff.codegen.workload import workload_procedures, wrap_module
from thermodiff.derivatives import resolve_derivatives
from thermodiff.diffplz import DiffPlz
from thermodiff.fortran import _sanitize, to_fortran
from thermodiff.model.errors import ModelError

__all__ = ["build_diff", "emit_module", "select_derivatives"]

#: Candidate names for the generated loop variables.  ``i`` and ``j`` are
#: intentionally absent: they are reserved for the derivative indices.
_LOOP_NAMES = list("klmpqrsuwxyzabcdeghov")


def select_derivatives(declared_variables):
    """Return every derivative applicable to the declared variables.

    Parameters
    ----------
    declared_variables : set of str
        Base names of the declared variables (``"n"``, ``"T"``, ``"V"``,
        ``"P"``).

    Returns
    -------
    list of str
        Attribute names understood by
        :meth:`thermodiff.diffplz.DiffPlz` and
        :func:`thermodiff.fortran.to_fortran`.
    """
    return resolve_derivatives(["all"], declared_variables)


def _reserved_names(ir, derivatives):
    """Names that the generated loop variables must avoid."""
    reserved = {"i", "j", "nc", "n", "f", "res", "T", "V", "P"}
    reserved.update(ir.imported)
    reserved.update(ir.dimensions)
    for declaration in list(ir.variables) + list(ir.parameters):
        reserved.add(declaration.base)
    reserved.update(placeholder.base for placeholder in ir.placeholders)
    for symbol in ir.function.atoms(sp.Symbol):
        reserved.add(str(symbol))
    for indexed in ir.function.atoms(sp.Indexed):
        reserved.add(str(indexed.base))
    return reserved


def _index_mapping(ir, derivatives):
    """Return the mapping from the internal indices to friendly ones."""
    reserved = _reserved_names(ir, derivatives)
    mapping = {}
    used = set()
    # Fortran is case insensitive.
    reserved = {name.lower() for name in reserved}
    for index in sorted(ir.indexes, key=str):
        for candidate in _LOOP_NAMES:
            if candidate not in reserved and candidate not in used:
                mapping[index] = sp.Idx(candidate)
                used.add(candidate)
                break
        else:
            raise ModelError("too many distinct indices in the model")
    return mapping


def _rename_indices(ir, derivatives):
    """Return ``(function, indexes)`` with friendly index names."""
    return _apply_mapping(ir, _index_mapping(ir, derivatives))


def _apply_mapping(ir, mapping):
    """Return ``(function, indexes)`` renamed with ``mapping``."""
    function = ir.function.xreplace(mapping)
    indexes = [mapping.get(index, index) for index in ir.indexes]
    return function, indexes


def build_diff(ir, derivatives=None):
    """Differentiate a :class:`ModelIR` with friendly index names.

    This is the step shared by Fortran emission and the LaTeX preview, so
    it is exposed on its own to avoid differentiating twice.

    Parameters
    ----------
    ir : ModelIR
        Symbolic model built by :mod:`thermodiff.model.ir`.
    derivatives : str or list of str, optional
        Derivatives to select.  Accepts the friendly names and groups of
        :func:`thermodiff.derivatives.resolve_derivatives`.  Defaults to
        :func:`select_derivatives`, that is, every derivative applicable
        to the declared variables.

    Returns
    -------
    diff : DiffPlz
        Differentiated model, with the user indices renamed to friendly
        loop variables.
    derivatives : list of str
        The derivative attributes that were selected.
    """
    if derivatives is None:
        derivatives = select_derivatives(ir.declared_variables)
    else:
        try:
            derivatives = resolve_derivatives(
                derivatives, ir.declared_variables
            )
        except ValueError as error:
            raise ModelError(str(error)) from error
    mapping = _index_mapping(ir, derivatives)
    function, indexes = _apply_mapping(ir, mapping)
    calls = [call.xreplace(mapping) for call in ir.placeholder_calls]
    diff = DiffPlz(
        function,
        internal_functions=calls,
        indexes=indexes,
        name="f",
        derivatives=derivatives,
    )
    return diff, derivatives


def _array_extents(ir):
    """Map each array name of the model to its dimension names."""
    declared = list(ir.variables) + list(ir.parameters)
    declared += list(ir.placeholders)
    return {item.base: tuple(item.dims) for item in declared if item.dims}


def _workload_code(
    ir,
    diff,
    workloads,
    fixed_nc,
    subroutine,
    preset,
    options,
    overrides,
    common,
):
    """Return the dispatcher and kernels of the workload mode."""
    try:
        selections = [
            resolve_derivatives(workload, ir.declared_variables)
            for workload in workloads
        ]
        return workload_procedures(
            diff,
            selections,
            subroutine_name=subroutine,
            fixed_nc=fixed_nc,
            extents=_array_extents(ir),
            preset=preset,
            options=options,
            **common,
            **overrides,
        )
    except ValueError as error:
        raise ModelError(str(error)) from error


def emit_module(
    ir,
    derivatives=None,
    diff=None,
    *,
    preset=None,
    options=None,
    workloads=None,
    fixed_nc=None,
    **overrides,
):
    """Generate the Fortran module for a :class:`ModelIR`.

    The code-generation options come from a preset (``"standard"`` by
    default), or from ``options``, with ``overrides`` applied on top.

    Parameters
    ----------
    ir : ModelIR
        Symbolic model built by :mod:`thermodiff.model.ir`.
    derivatives : str or list of str, optional
        Derivatives to emit.  Accepts the friendly names and groups of
        :func:`thermodiff.derivatives.resolve_derivatives`.  Defaults to
        :func:`select_derivatives`.
    diff : DiffPlz, optional
        A pre-built differentiation, as returned by :func:`build_diff`.
        When given, ``derivatives`` must match the ones it was built with.
    preset : {"minimal", "standard", "fastest"}, optional
        Named option set.  ``"minimal"`` is the plain generator (the output
        before presets existed), ``"standard"`` (the default) is the
        recommended set, and ``"fastest"`` gives the fastest generated code
        of the benchmark campaign at the cost of slower generation and
        last-bit rounding differences; it pays off with one kernel per
        derivative set (``optional_derivatives=False``).
    options : CodegenOptions, optional
        Explicit options used instead of a preset.  Passing both
        ``preset`` and ``options`` raises ``ValueError``.
    workloads : list of (str or list of str), optional
        Workload mode: one derivative selection per fixed-output kernel,
        for example ``["dn,dndn", "dT"]``.  The module then holds a
        dispatcher with the signature of the ``optional_derivatives=True``
        kernel (which is also emitted, as the fallback), and one kernel
        per workload; see :mod:`thermodiff.codegen.workload`.  ``derivatives``
        selects the outputs of the fallback and must cover the workloads.
    fixed_nc : iterable of int, optional
        With ``workloads``: also emit every fixed-output kernel for these
        compile-time numbers of components (for example ``(2, 3)``).
    **overrides
        Individual options replacing those of the preset, with the names of
        :class:`thermodiff.codegen.options.CodegenOptions` (for example
        ``optional_derivatives``, ``array_cse``, ``share``,
        ``param_symmetry``, ``cse_backend``).  ``None`` means "not given".
        ``stats`` is forwarded to :func:`thermodiff.fortran.to_fortran`.
        The symmetric parameters declared in the model are always
        forwarded.

    Returns
    -------
    str
        The complete Fortran module.
    """
    if diff is None:
        diff, derivatives = build_diff(ir, derivatives)
    elif derivatives is None:
        derivatives = getattr(diff, "derivatives", None)
        if derivatives is None:
            derivatives = select_derivatives(ir.declared_variables)

    stats = overrides.pop("stats", None)
    model = _sanitize(ir.model_name)
    subroutine = _sanitize(ir.model_name) + "_derivs"
    common = dict(
        derivatives=derivatives,
        name="f",
        integer_args=ir.dimensions,
        module_symbols=ir.imported,
        symmetric_params=getattr(ir, "symmetric", ()),
        stats=stats,
    )
    if workloads is None and fixed_nc is not None:
        raise ModelError("fixed_nc needs workloads")
    if workloads is not None:
        code = _workload_code(
            ir,
            diff,
            workloads,
            fixed_nc,
            subroutine,
            preset,
            options,
            overrides,
            common,
        )
    else:
        options = resolve_codegen_options(preset, options, **overrides)
        code = to_fortran(
            diff, subroutine_name=subroutine, options=options, **common
        )

    return wrap_module(
        f"{model}_mod",
        ir.uses,
        code,
        [
            f"! Generated by thermodiff from model {ir.model_name!r}.",
            "! Do not edit by hand.",
        ],
    )
