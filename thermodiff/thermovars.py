"""Thermodynamic variables and indices module.

This module defines the SymPy thermodynamic variables and indices used in
thermodynamic calculations. It uses sympy for symbolic mathematics.

Module variables
    - `nc`: Symbol representing the number of components (integer).
    - `i`, `j`, `k`, `l`, `m`: Indexed symbols for indices used in equations.
    - `n`: IndexedBase varaible for the number of moles of each component.
    - `T`: Symbol for temperature.
    - `V`: Symbol for volume.
    - `P`: Symbol for pressure.
    - `R`: Symbol for the gas constant.

You can directly import these variables by doing:

.. code-block:: python

    from thermodiff import k, l, m      # Indices
    from thermodiff import n, P, T, V   # Thermodynamic variables
    form thermodiff import R            # Gas constant

Those are the normal variables used in thermodynamic calculations. Never use
`i` and `j` directly. those are used for derivatives of `n`, and are not
intended for direct use in equations (See tutorial).
"""

import sympy as sp

# Variable dummy Nc (number of components)
nc = sp.symbols("N_c", integer=True)

# Indices
# For n derivatives, the user don't use these directly.,
i = sp.symbols("i", cls=sp.Idx)
j = sp.symbols("j", cls=sp.Idx)

# Index for variables and things
k = sp.symbols("k", cls=sp.Idx)
l = sp.symbols("l", cls=sp.Idx)
m = sp.symbols("m", cls=sp.Idx)

# Thermo variables
n = sp.IndexedBase("n", shape=(nc,))
T = sp.symbols("T")
V = sp.symbols("V")
P = sp.symbols("P")

# Gas constant
R = sp.symbols("R")
