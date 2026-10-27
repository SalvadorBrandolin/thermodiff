"""Bundled example models for the graphical front-end.

The GUI needs its presets to be available when ``thermodiff`` is installed
from a wheel, where the top level ``examples/`` directory is not shipped.
The models are therefore embedded here as plain strings.  The ones that also
exist as files in ``examples/`` are kept in sync by a test.

Use :func:`example_names` and :func:`get_example` instead of touching the
dictionaries directly.
"""

from collections import OrderedDict

__all__ = [
    "EXAMPLES",
    "DEFAULT_EXAMPLE",
    "example_names",
    "get_example",
]


#: Ordered mapping of example name -> ``.model`` file contents.
EXAMPLES = OrderedDict(
    [
        (
            "Peng-Robinson (pressure)",
            """\
model pr
target fortran
use yaeos__constants, only: R

variables
    n   nc
    V
    T

parameters
    ac    nc
    Tc    nc
    k     nc nc
    b     nc

expressions
    alpha_i = (T/Tc_i)^2
    a_i     = ac_i*alpha_i
    amix    = sum_i sum_j n_i*n_j*sqrt(a_i*a_j)*(1 - k_ij)
    bmix    = sum_i n_i*b_i

function
    R*T/(V - bmix) - amix/(V*(V + bmix) + bmix*(V - bmix))
""",
        ),
        (
            "Ideal gas enthalpy",
            """\
model ideal_enthalpy
target fortran

variables
    n   nc
    T

parameters
    cp    nc
    Tref

expressions
    h_i = cp_i*(T - Tref)

function
    sum_i n_i*h_i
""",
        ),
        (
            "Mole fraction",
            """\
model molefrac
target fortran

variables
    n   nc

expressions
    ntot = sum_i n_i

function
    n_k / ntot
""",
        ),
        (
            "Minimal (edit me)",
            """\
model my_model
target fortran

variables
    n   nc
    T

function
    sum_i n_i*T
""",
        ),
    ]
)

#: Example loaded when the GUI starts.
DEFAULT_EXAMPLE = "Peng-Robinson (pressure)"


def example_names():
    """Return the example names in display order.

    Returns
    -------
    list of str
        Names accepted by :func:`get_example`.
    """
    return list(EXAMPLES)


def get_example(name):
    """Return the ``.model`` text of an example.

    Parameters
    ----------
    name : str
        One of :func:`example_names`.  Unknown names fall back to the
        default example.

    Returns
    -------
    str
        The ``.model`` file contents.
    """
    return EXAMPLES.get(name, EXAMPLES[DEFAULT_EXAMPLE])
