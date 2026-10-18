Model files (``.model``)
========================

The ``.model`` front-end lets you describe a thermodynamic model in a small,
declarative text format and compile it into a self contained Fortran module
with the function and all its analytical derivatives.

It is designed to be the *single entry point* for users that do not want to
write Python: you write a ``.model`` file and run::

    thermodiff-model model.model -o model.f90

or, without installing the console script::

    python -m thermodiff.model model.model -o model.f90

If you would rather click than type, an optional
:doc:`graphical front-end <gui>` wraps the same compiler.

File format
-----------

A ``.model`` file is organized in sections.  Section headers start at the
beginning of a line:

``model``
    Line ``model <name>``.  Gives the module name (``<name>_mod``) and the
    subroutine name (``<name>_derivs``).

``target``
    Line ``target fortran``.  Only ``fortran`` is supported.

``use``
    One line per Fortran ``use`` statement, for example
    ``use yaeos__constants, only: R``.  Names listed after ``only:`` become
    known symbols of the model (they are not generated as arguments).

``variables``
    Declarations of the independent variables.  Names ``n``, ``V``, ``T``
    and ``P`` are recognized by the derivative engine; any other name is
    treated as an extra input.

``parameters``
    Declarations of the parameters.

``expressions``
    Named intermediate expressions.  They are substituted into the final
    function before differentiation, so they never need to be evaluated by
    the generated code.

``function``
    The expression to differentiate, written exactly as it should be
    differentiated.

Declarations
------------

A declaration is ``name`` optionally followed by one dimension per rank::

    n    nc
    V
    T

    k    nc nc
    b    nc

Dimensions are integer arguments of the generated subroutine.  The name
``nc`` is available by default and is also used as the default upper limit of
summations.  Extra dimensions (for example ``ncomp``) become additional
``integer`` arguments.

A trailing ``_letters`` in a declaration is treated as a decorative
subscript and only refines the base name; for example ``k_ij nc nc`` declares
the rank two array ``k``.

Expressions
-----------

Expressions support the usual arithmetic operators ``+ - * /``, powers written
``^`` or ``**``, implicit multiplication (``2 x``, ``n_i b_i``), a set of
built-in functions (``sqrt``, ``exp``, ``log``, ``sin``, ``min``, ...) and
indexed names.

An indexed name is written ``base_indices`` where ``base`` is a declared name
and ``indices`` is a string of single letters::

    n_i          # n[i]
    k_ij         # k[i, j]

A definition with a free index is an array; a definition without free indices
is a scalar.  Every free index of the right hand side must be balanced by the
subscripts of the left hand side, and every summation index must be bound by a
``sum``.

Summations
----------

The summation operator is written with an ASCII syntax::

    sum_i <expression>              # i from 1 to nc
    sum_i[a:b] <expression>         # i from a to b
    sum_i[a:] <expression>          # i from a to nc
    sum_i[:b] <expression>          # i from 1 to b

The body of a sum is the following product, which makes the common
``sum_i n_i * b_i`` read naturally.  Nested sums are allowed.

Validation
----------

The compiler is deliberately strict so that mistakes are caught early.  All
of the following are hard errors with the offending file and line:

* undefined names,
* a subscript count that does not match the declared rank,
* unbalanced free indices,
* an unknown dimension or summation bound,
* reusing an index in nested sums,
* duplicate declarations,
* unknown sections or malformed declarations.

Selecting derivatives
---------------------

By default the compiler emits the function and every first and second order
derivative among the declared variables.  When only a few of them are needed,
request them explicitly and the generated subroutine declares just those
outputs::

    thermodiff-model model.model --derivatives dn,dndn

The names are the generated output variables without the leading ``df``:
``dT``, ``dV``, ``dP``, ``dn``, ``dT2``, ``dV2``, ``dP2``, ``dndn``,
``dTdV``, ``dTdP``, ``dVdP``, ``dndT``, ``dndV`` and ``dndP``.  A leading
``df`` is optional, the order of the mixed derivatives is free (``dTdn`` and
``dndT`` are the same) and the :class:`~thermodiff.diffplz.DiffPlz` attribute
names (``dt``, ``dni`` ...) are accepted as aliases.

The groups ``all`` (the default), ``none`` (only the function value) and
``first`` / ``second`` select several derivatives at once::

    thermodiff-model model.model --derivatives first

Requesting a derivative that does not apply to the declared variables is an
error; for example ``dP`` requires a variable named ``P``.

Computing derivatives on demand
-------------------------------

Selecting derivatives decides which outputs *exist*; the caller still has to
pass every one of them.  When the number of derivatives grows this becomes
wasteful, because a given caller usually needs only a handful.

With ``--optional-derivatives`` every output - the function value included -
is declared as a Fortran ``optional`` argument, and each output (together
with the shared terms that only it needs) is computed only when the caller
actually passes it::

    thermodiff-model model.model --optional-derivatives

The generated subroutine can then be called with any subset of the
derivatives, in any order, using keyword arguments::

    call pr_derivs(nc, V, T, Tc, ac, b, k, n, f=f)          ! only the value
    call pr_derivs(nc, V, T, Tc, ac, b, k, n, dfdT=dfdT)    ! only dT
    call pr_derivs(nc, V, T, Tc, ac, b, k, n, f=f, dfdn=dfdn, dfdndn=dfdndn)

Computations that several outputs share are guarded by the *union* of their
``present`` conditions, so a summation is skipped unless at least one of the
outputs that needs it was passed.  This mirrors the style used throughout
`yaeos <https://github.com/ipqa-research/yaeos>`_.

Shared computations
-------------------

The generated code evaluates every distinct summation once and reuses it
across the derivatives that need it.  Sub-expressions that do not depend on
an index and are used by several derivatives are then hoisted into common
temporaries.  For the Peng-Robinson example above the mixture parameters
``amix`` and ``bmix`` - and the sums they are built from - are therefore
computed a single time instead of once per derivative.

Python API
----------

.. automodule:: thermodiff.model
   :members:
   :undoc-members:
   :show-inheritance:
