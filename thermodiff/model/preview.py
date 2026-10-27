"""Readable LaTeX preview of a differentiated model.

This is a thin, GUI friendly layer on top of
:meth:`thermodiff.diffplz.DiffPlz.latex_readable_plz`.  It reuses the same
:class:`~thermodiff.diffplz.DiffPlz` instance that Fortran emission builds,
so it does not differentiate anything twice.
"""

import sympy as sp

from thermodiff.fortran import _DERIV_SUFFIX
from thermodiff.latex_utils import clean_latex_string

__all__ = ["latex_preview"]


#: ``DiffPlz`` derivative attribute -> key returned by
#: :meth:`DiffPlz.latex_readable_plz`.
_LATEX_KEY = {
    "dt": "dT",
    "dv": "dV",
    "dp": "dP",
    "dni": "dni",
    "dt2": "dT2",
    "dv2": "dV2",
    "dp2": "dP2",
    "dnidnj": "dn2",
    "dtdv": "dTV",
    "dtdp": "dTP",
    "dvdp": "dVP",
    "dtdni": "dTn",
    "dvdni": "dVn",
    "dpdni": "dPn",
}

#: LaTeX title shown for each derivative.
_LATEX_TITLE = {
    "dt": r"\frac{\partial f}{\partial T}",
    "dv": r"\frac{\partial f}{\partial V}",
    "dp": r"\frac{\partial f}{\partial P}",
    "dni": r"\frac{\partial f}{\partial n_i}",
    "dt2": r"\frac{\partial^2 f}{\partial T^2}",
    "dv2": r"\frac{\partial^2 f}{\partial V^2}",
    "dp2": r"\frac{\partial^2 f}{\partial P^2}",
    "dnidnj": r"\frac{\partial^2 f}{\partial n_i \partial n_j}",
    "dtdv": r"\frac{\partial^2 f}{\partial T \partial V}",
    "dtdp": r"\frac{\partial^2 f}{\partial T \partial P}",
    "dvdp": r"\frac{\partial^2 f}{\partial V \partial P}",
    "dtdni": r"\frac{\partial^2 f}{\partial T \partial n_i}",
    "dvdni": r"\frac{\partial^2 f}{\partial V \partial n_i}",
    "dpdni": r"\frac{\partial^2 f}{\partial P \partial n_i}",
}


def latex_preview(diff, derivatives):
    """Return a LaTeX preview of the function and its derivatives.

    Parameters
    ----------
    diff : DiffPlz
        Differentiated model, as returned by
        :func:`thermodiff.model.emit.build_diff`.
    derivatives : list of str
        Derivative attributes to show, in order.  Attributes that are not
        part of the LaTeX output are skipped.

    Returns
    -------
    dict
        ``{"function": str, "derivatives": [ {"key", "title", "name",
        "latex"} ]}`` where ``latex`` is a LaTeX math string and ``name``
        is the corresponding Fortran output variable name.
    """
    values = diff.latex_readable_plz()

    entries = []
    for attribute in derivatives:
        key = _LATEX_KEY.get(attribute)
        if key is None or key not in values:
            continue
        entries.append(
            {
                "key": key,
                "title": _LATEX_TITLE.get(attribute, key),
                "name": "df" + _DERIV_SUFFIX[attribute],
                "latex": values[key],
            }
        )

    function = clean_latex_string(sp.latex(diff.expression))
    return {"function": function, "derivatives": entries}
