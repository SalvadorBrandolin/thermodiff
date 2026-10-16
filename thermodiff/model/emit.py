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

from thermodiff.diffplz import DiffPlz
from thermodiff.fortran import _sanitize, to_fortran
from thermodiff.model.errors import ModelError

__all__ = ["build_diff", "emit_module", "select_derivatives"]

#: Candidate names for the generated loop variables.  ``i`` and ``j`` are
#: intentionally absent: they are reserved for the derivative indices.
_LOOP_NAMES = list("klmpqrsuwxyzabcdeghov")


def select_derivatives(declared_variables):
    """Return the derivative attributes to emit.

    All first and second order derivatives among the declared variables
    are selected (they are still skipped if they are identically zero).

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
    declared = set(declared_variables)
    selected = []
    if "T" in declared:
        selected += ["dt", "dt2"]
    if "V" in declared:
        selected += ["dv", "dv2"]
    if "P" in declared:
        selected += ["dp", "dp2"]
    if "n" in declared:
        selected += ["dni", "dnidnj"]
    if {"T", "V"} <= declared:
        selected.append("dtdv")
    if {"T", "P"} <= declared:
        selected.append("dtdp")
    if {"V", "P"} <= declared:
        selected.append("dvdp")
    if {"T", "n"} <= declared:
        selected.append("dtdni")
    if {"V", "n"} <= declared:
        selected.append("dvdni")
    if {"P", "n"} <= declared:
        selected.append("dpdni")
    return selected


def _reserved_names(ir, derivatives):
    """Names that the generated loop variables must avoid."""
    reserved = {"i", "j", "nc", "n", "f", "res", "T", "V", "P"}
    reserved.update(ir.imported)
    reserved.update(ir.dimensions)
    for declaration in list(ir.variables) + list(ir.parameters):
        reserved.add(declaration.base)
    for symbol in ir.function.atoms(sp.Symbol):
        reserved.add(str(symbol))
    for indexed in ir.function.atoms(sp.Indexed):
        reserved.add(str(indexed.base))
    return reserved


def _rename_indices(ir, derivatives):
    """Return ``(function, indexes)`` with friendly index names."""
    reserved = _reserved_names(ir, derivatives)
    mapping = {}
    used = set()
    for index in sorted(ir.indexes, key=str):
        for candidate in _LOOP_NAMES:
            if candidate not in reserved and candidate not in used:
                mapping[index] = sp.Idx(candidate)
                used.add(candidate)
                break
        else:
            raise ModelError("too many distinct indices in the model")
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
    derivatives : list of str, optional
        Derivative attributes to select.  Defaults to
        :func:`select_derivatives`.

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
    function, indexes = _rename_indices(ir, derivatives)
    diff = DiffPlz(function, internal_functions=[], indexes=indexes, name="f")
    return diff, derivatives


def emit_module(ir, derivatives=None, diff=None):
    """Generate the Fortran module for a :class:`ModelIR`.

    Parameters
    ----------
    ir : ModelIR
        Symbolic model built by :mod:`thermodiff.model.ir`.
    derivatives : list of str, optional
        Derivative attributes to emit.  Defaults to
        :func:`select_derivatives`.
    diff : DiffPlz, optional
        A pre-built differentiation, as returned by :func:`build_diff`.
        When given, ``derivatives`` must match the ones it was built with.

    Returns
    -------
    str
        The complete Fortran module.
    """
    if diff is None:
        diff, derivatives = build_diff(ir, derivatives)
    elif derivatives is None:
        derivatives = select_derivatives(ir.declared_variables)

    model = _sanitize(ir.model_name)
    subroutine = _sanitize(ir.model_name) + "_derivs"

    code = to_fortran(
        diff,
        derivatives=derivatives,
        name="f",
        subroutine_name=subroutine,
        integer_args=ir.dimensions,
        module_symbols=ir.imported,
    )

    lines = [
        f"! Generated by thermodiff from model {ir.model_name!r}.",
        "! Do not edit by hand.",
        f"module {model}_mod",
    ]
    for use_text in ir.uses:
        lines.append(f"   use {use_text}")
    lines.extend(["   implicit none", "contains", ""])
    for line in code.splitlines():
        lines.append("   " + line if line.strip() else "")
    lines.append(f"end module {model}_mod")
    lines.append("")
    return "\n".join(lines)
