"""Fortran code generation for thermodynamic models.

This module turns a :class:`~thermodiff.diffplz.DiffPlz` instance into a
free-form Fortran subroutine that evaluates the original function together
with all its available derivatives.

The generated subroutine takes the independent variables (mole numbers
``n``, volume ``V``, pressure ``P``, temperature ``T``), the model
parameters, and returns the function value and the derivatives as ``intent
(out)`` arguments, for example::

    subroutine f(n, V, T, f, dfdT, dfdV, dfdn)
      implicit none
      integer, intent(in) :: nc
      real(8), intent(in) :: n(:)
      real(8), intent(in) :: V, T
      real(8), intent(out) :: f, dfdT, dfdV
      real(8), intent(out) :: dfdn(:)
      ...
    end subroutine

Summations produced with :func:`~thermodiff.core.easy_sums.sum_components`
are expanded into explicit ``do`` loops and the Kronecker deltas handled by
``DiffPlz`` become ``if``/``else`` blocks.  Internal functions created with
:func:`~thermodiff.core.idxfunction.idx_function` (and their derivatives)
are exposed as extra input arguments by default; pass ``strict=True`` to
raise an error instead.
"""

from thermodiff.codegen.common import (  # noqa: F401
    _DERIV_SUFFIX,
    _Namer,
    _free_indices,
    _index_set,
    _sanitize,
)
from thermodiff.codegen.core_chain import (  # noqa: F401
    scalar_chain_core_derivatives,
    scalar_chain_routine,
)
from thermodiff.codegen.cse import (  # noqa: F401
    _best_indexed_candidate,
    _best_scalar_candidate,
    _common_scalar_cse,
    _indexed_cse,
    _preorder_entries,
    _sorted_sums,
    _sum_atoms,
)
from thermodiff.codegen.loops import _nest_double_sums  # noqa: F401
from thermodiff.codegen.pipeline import (  # noqa: F401
    _chain_structure,
    _emit_code,
    to_fortran,
)
from thermodiff.codegen.render import _expand_sums  # noqa: F401
from thermodiff.codegen.resolve import (  # noqa: F401
    _derivative_function_atoms,
    _scan_atoms,
)
from thermodiff.codegen.symmetry import (  # noqa: F401
    _canonical_symmetric,
    _fold_symmetric_sums,
    _swap,
)
from thermodiff.codegen.workload import workload_module

__all__ = ["to_fortran", "workload_module"]
