Fortran code generation
=======================

.. automodule:: thermodiff.fortran
   :members:
   :undoc-members:
   :show-inheritance:

Code generator options
----------------------

.. automodule:: thermodiff.codegen.options
   :members: CodegenOptions, Preset, ScalarChainMode, SepSumsMode,
             LowrankHessMode, CseBackend, PRESET_SETTINGS,
             resolve_codegen_options
   :show-inheritance:

Extra scalar variables: the scalar core recipe
----------------------------------------------

Besides ``T``, ``V``, ``P`` and ``n[i]``, a ``DiffPlz`` can differentiate
with respect to extra scalar symbols given in ``variables``.  This
supports models written as a small *scalar core* ``F(B, D, D1, nT, V, T)``
(a cubic equation of state in the mixture level covolume ``B``,
attraction ``D`` ...) whose arguments depend on the composition through
mixing rules.  The generator differentiates the core with respect to its
scalars and the chain rule over the composition is assembled by a generated
routine, which gives the hand-factored cost structure.

.. code-block:: python

    import sympy as sp
    import thermodiff as td

    B, D, D1, nT = sp.symbols("B D D1 nT")
    core = -nT * sp.log(1 - B / td.V) - D / (D1 * td.T) * sp.log(...)
    diff = td.DiffPlz(core, name="f", variables=[B, D, D1, nT])
    code = td.to_fortran(diff)  # every non-zero derivative, or select:
    code = td.to_fortran(diff, derivatives=["dB", "dBdD", "dBdT", "dV"])

Names are those of the thermodynamic derivatives with the variable name:
``dB``, ``dB2``, ``dBdD``, mixed with the thermodynamic variables
``dBdT``, ``dBdV``, ``dBdP``, and ``dndB`` (an array over ``n[i]`` when
the expression depends on ``n``).  Within a mixed derivative the order is
``n``, the declared variables in declaration order, then ``T``, ``V``,
``P`` (other orders are accepted).  The outputs are ``dfdB``,
``dfdBdD``, ``dfdndB`` ... and the variables are ``real(8),
intent(in)`` scalar arguments.  Variable names are case sensitive and
cannot be ``T``, ``V``, ``P`` or ``n``.  The groups ``first``, ``second``
and ``all`` include them (and, as always, emit zero outputs for the
variables the expression lacks).  With ``derivatives=None`` every
derivative, the scalar ones included, is computed and the non-zero ones
are emitted; without ``variables`` nothing changes.  The derivatives are also stored in
``DiffPlz.scalar_derivatives`` (keys such as ``"var_dBdD"``).

With mixing rules ``x_r(n)`` for the scalars ``x_r`` (``B = b.n``,
``D = n.a.n``, ``nT = sum(n)`` ...), the composition derivatives follow
from the chain rule:

.. math::

    \frac{\partial F}{\partial n_i} = \sum_r F_{x_r}\,\frac{\partial x_r}{\partial n_i},
    \qquad
    \frac{\partial^2 F}{\partial n_i \partial n_j} =
    \sum_{r,s} F_{x_r x_s}\,x_{r,i}\,x_{s,j} + \sum_r F_{x_r}\,x_{r,ij}

and likewise ``F_{T n_i} = sum_r F_{T x_r} x_{r,i}`` (``dBdT`` ... with
``T`` and ``V``).  Scalars may also depend on ``T`` and ``V`` (``D(n, T)``):
the routine below includes those terms.

The assembly routine
~~~~~~~~~~~~~~~~~~~~

:func:`thermodiff.scalar_chain_routine` emits that step as a Fortran
routine, so no hand-written assembly is left.  Declare the scalars in the
order of ``variables``, which of ``T`` and ``V`` each one depends on (all
depend on ``n``) and which are linear in ``n`` (no Hessian input):

.. code-block:: python

    names = ["B", "D", "D1", "nT"]
    kwargs = dict(
        linear=["B", "D1", "nT"],       # d2x/dn dn = 0 for these
        derivatives=["dT", "dV", "dn", "dndT", "dndV", "dndn"],
    )
    # D = n.a.n * alpha(T) would be depends_on={"D": ["T"]}
    wanted = td.scalar_chain_core_derivatives(names, {}, **kwargs)
    core_code = td.to_fortran(diff, derivatives=wanted)
    chain_code = td.scalar_chain_routine(names, {}, **kwargs)

``scalar_chain_core_derivatives`` lists the derivatives to request from the
core (``dB``, ``dBdD``, ``dBdT``, ``dT`` ...), and the routine reads exactly
those as inputs named as the core kernel emits them (``dfdB``, ``dfdBdD``,
``dfdT2`` ...; ``core_name`` changes the prefix).  Per scalar ``X`` the
inputs are its gradient ``dXdn(nc)`` and Hessian ``dXdndn(nc, nc)``, and if
it depends on ``T``: ``dXdT``, ``dXdT2``, ``dXdndT(nc)`` (``dXdTdV`` when it
depends on both).  The outputs are the total derivatives with the usual
names and prefix ``dar`` (``name``): ``dardT``, ``dardV``, ``dardT2``,
``dardTdV``, ``dardV2``, ``dardn``, ``dardndT``, ``dardndV``, ``dardndn``,
selected by ``derivatives`` like in ``to_fortran``.  With
``CodegenOptions(optional_derivatives=True)`` every output and input is
``optional`` and an output is computed when present (give its inputs);
``real_kind`` and ``kind_module`` are honoured.  The Hessian loop works on the
lower triangle and mirrors it, with ``O(m)`` work per element for ``m``
scalars.  ``tests/test_scalar_variables.py`` compares the core kernel plus
this routine with the kernel written directly in the mole numbers.

.. autofunction:: thermodiff.codegen.core_chain.scalar_chain_routine

.. autofunction:: thermodiff.codegen.core_chain.scalar_chain_core_derivatives

Placeholders with expression arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A placeholder whose argument is an expression, for example
``alpha(k, delta(n, V), tau(n, T))`` in a multifluid model, is treated as
a function of the variables the expression contains, and the generated
kernel asks the caller for its *total* derivatives with respect to them
(``dalphadn``, ``dalphadV`` ...).  The generator does not apply the chain
rule through the arguments, and emits a ``UserWarning`` naming the
placeholder and the expression.  To get the chain rule through ``delta``
and ``tau``, make them scalar variables of the placeholder's core and use
the recipe above: generate the mixture layer with ``variables=[delta,
tau]`` written as symbols, supply ``alpha`` and its partial derivatives
with respect to them, and assemble.
Workload mode: fixed-output kernels and a dispatcher
----------------------------------------------------

``optional_derivatives=True`` gives one kernel for any output set, but it
pays a fixed cost on every call and disables the loop passes of the
``fastest`` preset.  The workload mode keeps that single public procedure
and makes it cheap.  From the command line::

    thermodiff-model nrtl.model --preset fastest \
        --workload dn,dndn --workload dT --derivatives dT,dn,dndn

or from Python, ``compile_model(text, workloads=["dn,dndn", "dT"])`` and
:func:`thermodiff.workload_module` for a ``DiffPlz`` object.  The module
contains

* a dispatcher ``<model>_derivs`` with the signature of the optional
  kernel; it builds a bit mask from ``present()`` and selects, with
  ``select case``, the kernel whose output set is exactly the present
  set;
* one fixed-output kernel ``<model>_derivs_w<k>`` per workload, generated
  with ``optional_derivatives=False`` and the chosen preset (explicit-shape
  dummies, ``!GCC$ ATTRIBUTES NOINLINE``);
* the optional kernel ``<model>_derivs_fallback``, used for every other
  combination of present outputs.  ``--derivatives`` selects its outputs and
  must cover the workloads.

A workload is matched by the output set of its kernel, including the
function value when the kernel computes it.  A call with a larger set is not
served by a covering kernel (that would need scratch arrays for the extra
outputs); it goes to the fallback.

``--fixed-nc 2,3`` (``fixed_nc=(2, 3)``) also emits, for each workload and
each size, a copy ``<model>_derivs_w<k>_nc<size>`` where ``nc`` is a named
constant (the dummy ``nc_in`` keeps the signature).  The compiler then
unrolls the loops, drops the array descriptors and keeps the temporaries on
the stack.  The dispatcher selects by ``nc`` and then by output set.

Pitfalls handled by the generator (measured with gfortran 16): the
dispatcher's optional array dummies are not ``contiguous`` (an absent
actual argument could be packed, and crash), while its inputs are; the
kernels have explicit-shape dummies so that forwarding makes no copy;
and the kernels are not inlined into the dispatcher, which would make them
lose the contiguity facts.  Dimensions other than ``nc`` (for example
``ng``) are forwarded from the kernels' own dummy lists; their arrays keep
assumed shape in the kernels unless the extents are known (the ``.model``
front end passes them).  Explicit shape re-strides an actual array whose
leading extents differ from the declared ones (a ``(3, 3)`` array passed with
``nc = 2``), so the dispatcher calls a fixed-output kernel only when every
array of rank two or more, input or output, has exactly the declared extents
in all dimensions but the last; otherwise it calls the fallback, which is
always correct.

.. automodule:: thermodiff.codegen.workload
   :members: workload_module, workload_procedures
   :show-inheritance:
