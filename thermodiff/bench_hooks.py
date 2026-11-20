"""Measurement hooks for the benchmark's generation phase.

This module exposes the callable API used by the session-root driver
(``scripts/generate-symbolic-variants.py``) and by ``benchmark-engineer``
to record, per (workload x variant):

* the wall time of generation and of its sub-phases (parse/IR,
  differentiation, Fortran emission),
* the path, line count and byte count of the generated file,
* the symbolic operation count of the final expressions (supplementary
  metadata; it is not a runtime predictor).

Every benchmark generation uses ``optional_derivatives=True`` (D-014), so
it is the default here.  The generated files are deterministic: identical
inputs give byte-identical output; no timestamps are ever written into a
generated file.  The ``use`` statements declared by the ``.model`` file are
spliced at generation time as the first body lines of the emitted bare
subroutine (so each file compiles standalone; see the interface-contract
handoff, producer side, route (a)).

Variant settings (D-007; ``reports/protocol.md`` section 13):

===========  ==========  =======  ==================================
variant      array_cse   share    passes run
===========  ==========  =======  ==================================
v1 "none"    False       False    none
v2 "partial" False       True     _share_sums + _common_scalar_cse
v3 "full"    True        True     _indexed_cse + the two above
===========  ==========  =======  ==================================
"""

import os
import time
import warnings

from thermodiff.codegen.options import CodegenOptions
from thermodiff.fortran import to_fortran
from thermodiff.model.emit import build_diff
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

__all__ = [
    "LEGACY_KEYS",
    "STAGE_VARIANTS",
    "VARIANTS",
    "WORKLOADS",
    "generate_workload",
    "variant_settings",
]

#: Benchmark generation variants (D-007).  ``key`` is the short name used
#: in file and routine names (``pr76_v1_w0``), ``token`` the directory
#: infix (``pr76_v1_none``) and ``id`` the method id of the protocol.
VARIANTS = {
    "v1": {
        "key": "v1",
        "id": "td_v1_none",
        "token": "none",
        "label": "V1 none",
        "array_cse": False,
        "share": False,
    },
    "v2": {
        "key": "v2",
        "id": "td_v2_partial",
        "token": "partial",
        "label": "V2 partial",
        "array_cse": False,
        "share": True,
    },
    "v3": {
        "key": "v3",
        "id": "td_v3_full",
        "token": "full",
        "label": "V3 full",
        "array_cse": True,
        "share": True,
    },
}

#: Optional codegen keys a variant may set (see ``to_fortran``).
_EXTRA_KEYS = (
    "share_sums",
    "scalar_cse",
    "param_symmetry",
    "hessian_symmetry",
    "cse_backend",
    "sum_alpha",
    "array_pad",
    "hoist_invariants",
    "chain_rule",
    "hoist_loops",
    "sum_free_alpha",
    "fuse_temps",
    "nest_sums",
    "split_sqrt",
    "split_sums",
    "nest_shared",
    "sym_matvec",
    "col_major",
    "pair_cse",
    "pair_rates",
    "sum_transpose",
    "sep_sums",
    "sym_rank3",
    "nest_general",
    "sym_sum2",
    "cse_expensive",
    "fuse_loops",
    "nested_nodes",
    "sym_perm",
    "fuse_guard",
    "nest_cost",
    "comp_fuse",
    "fuse_outer",
    "diag_table",
    "sib_fuse",
    "dummy_canon",
    "float_canon",
    "contiguous_args",
    "fuse_libm_guard",
    "keep_libm_tables",
    "table_layout",
    "lowrank_hess",
    "drop_pair_tables",
    "share_elementwise",
    "scalar_chain",
    "split_affine",
    "split_acc",
)

#: Extended variants for the ablation / symmetry study (thesis chapter).
#: Each one changes a single setting relative to its neighbour.
#:   a0  nothing                 (= v1)
#:   a1  + share_sums
#:   a2  + scalar_cse            (= v2)
#:   a3  + array_cse             (= v3)
#:   a4  a3 + hessian_symmetry
#:   a5  a3 + param_symmetry
#:   a6  a3 + both symmetries
#:   a7  a3 with the SymPy CSE backend
#:   a8  scalar_cse only (no share_sums, no array_cse)
#:   a9  a3 + alpha-equivalent summation sharing
#:   a10 a6 + alpha-equivalent summation sharing
EXTENDED_VARIANTS = {
    "a0": dict(array_cse=False, share=False),
    "a1": dict(array_cse=False, share=True, scalar_cse=False),
    "a2": dict(array_cse=False, share=True),
    "a3": dict(array_cse=True, share=True),
    "a4": dict(array_cse=True, share=True, hessian_symmetry=True),
    "a5": dict(array_cse=True, share=True, param_symmetry=True),
    "a6": dict(
        array_cse=True, share=True, param_symmetry=True, hessian_symmetry=True
    ),
    "a7": dict(array_cse=True, share=True, cse_backend="sympy"),
    "a8": dict(array_cse=False, share=False, scalar_cse=True),
    # added after the first campaign: alpha-equivalent summation sharing
    "a9": dict(array_cse=True, share=True, sum_alpha=True),
    "a10": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
    ),
    # added after the N >= 45 bimodality was found: padded rank-2 temporaries
    "a3p": dict(array_cse=True, share=True, array_pad=8),
    "a6p": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        array_pad=8,
    ),
    "a10p": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        array_pad=8,
    ),
    # added to close the RKPR regression: loop-invariant hoisting
    "a3h": dict(array_cse=True, share=True, hoist_invariants=True),
    "a6h": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        hoist_invariants=True,
    ),
    "a10h": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
    ),
    # chain-rule differentiation over summation nodes (structure of the
    # hand-written analytic code)
    "c3": dict(
        array_cse=True, share=True, hoist_invariants=True, chain_rule=True
    ),
    "c6": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
    ),
    "c6l": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
    ),
    "c6f": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
    ),
    "c6n": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
    ),
    "c6s": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        cse_backend="sympy",
    ),
    # closer to a careful hand-written implementation
    "c6x": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
    ),
    "c6r": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
    ),
    "c6m": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
    ),
    # --- generator-improvement iterations (each adds one option to c6m) ---
    "it01": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
    ),
    "it02": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
    ),
    "it03": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
    ),
    "it04": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
    ),
    "it05": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums=True,
    ),
    "it06": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums=True,
        sym_rank3=True,
    ),
    "it07": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sym_rank3=True,
    ),
    "it08": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums="cost",
        sym_rank3=True,
    ),
    "it09": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums="cost",
        sym_rank3=True,
        nest_general=True,
    ),
    "it10": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums="cost",
        sym_rank3=True,
        nest_general=True,
        sym_sum2=True,
    ),
    "it11": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums="cost",
        sym_rank3=True,
        nest_general=True,
        sym_sum2=True,
        cse_expensive=True,
    ),
    "it12": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums="cost",
        sym_rank3=True,
        nest_general=True,
        sym_sum2=True,
        cse_expensive=True,
        fuse_loops=True,
    ),
    "it13": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums="cost",
        sym_rank3=True,
        nest_general=True,
        sym_sum2=True,
        cse_expensive=True,
        fuse_loops=True,
        nested_nodes=True,
    ),
    "it14": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums="cost",
        sym_rank3=True,
        nest_general=True,
        sym_sum2=True,
        cse_expensive=True,
        fuse_loops=True,
        nested_nodes=True,
        sym_perm=True,
    ),
    "it15": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums="cost",
        sym_rank3=True,
        nest_general=True,
        sym_sum2=True,
        cse_expensive=True,
        fuse_loops=True,
        nested_nodes=True,
        sym_perm=True,
        fuse_guard=True,
    ),
    "it16": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums="cost",
        sym_rank3=True,
        nest_general=True,
        sym_sum2=True,
        cse_expensive=True,
        fuse_loops=True,
        nested_nodes=True,
        sym_perm=True,
        fuse_guard=True,
        nest_cost=True,
    ),
    "it17": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums="cost",
        sym_rank3=True,
        nest_general=True,
        sym_sum2=True,
        cse_expensive=True,
        fuse_loops=True,
        nested_nodes=True,
        sym_perm=True,
        fuse_guard=True,
        nest_cost=True,
        comp_fuse=True,
    ),
    "it18": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums="cost",
        sym_rank3=True,
        nest_general=True,
        sym_sum2=True,
        cse_expensive=True,
        fuse_loops=True,
        nested_nodes=True,
        sym_perm=True,
        fuse_guard=True,
        nest_cost=True,
        comp_fuse=True,
        fuse_outer=True,
    ),
    "it19": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums="cost",
        sym_rank3=True,
        nest_general=True,
        sym_sum2=True,
        cse_expensive=True,
        fuse_loops=True,
        nested_nodes=True,
        sym_perm=True,
        fuse_guard=True,
        nest_cost=True,
        comp_fuse=True,
        fuse_outer=True,
        diag_table=True,
    ),
    "it20": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums="cost",
        sym_rank3=True,
        nest_general=True,
        sym_sum2=True,
        cse_expensive=True,
        fuse_loops=True,
        nested_nodes=True,
        sym_perm=True,
        fuse_guard=True,
        nest_cost=True,
        comp_fuse=True,
        fuse_outer=True,
        diag_table=True,
        sib_fuse=True,
    ),
    "it21": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums="cost",
        sym_rank3=True,
        nest_general=True,
        sym_sum2=True,
        cse_expensive=True,
        fuse_loops=True,
        nested_nodes=True,
        sym_perm=True,
        fuse_guard=True,
        nest_cost=True,
        comp_fuse=True,
        fuse_outer=True,
        diag_table=True,
        sib_fuse=True,
        dummy_canon=True,
    ),
    "it22": dict(
        array_cse=True,
        share=True,
        param_symmetry=True,
        hessian_symmetry=True,
        sum_alpha=True,
        hoist_invariants=True,
        chain_rule=True,
        hoist_loops=True,
        sum_free_alpha=True,
        fuse_temps=True,
        nest_sums=True,
        split_sums=True,
        split_sqrt=True,
        nest_shared=True,
        sym_matvec=True,
        col_major=True,
        pair_cse=True,
        pair_rates=True,
        sum_transpose=True,
        sep_sums="cost",
        sym_rank3=True,
        nest_general=True,
        sym_sum2=True,
        cse_expensive=True,
        fuse_loops=True,
        nested_nodes=True,
        sym_perm=True,
        fuse_guard=True,
        nest_cost=True,
        comp_fuse=True,
        fuse_outer=True,
        diag_table=True,
        sib_fuse=True,
        dummy_canon=True,
        float_canon=True,
    ),
}
# Second optimisation round (reportes/brechas/ronda2-*.md); each step adds one option.
EXTENDED_VARIANTS["it23"] = dict(
    EXTENDED_VARIANTS["it22"], contiguous_args=True
)
EXTENDED_VARIANTS["it24"] = dict(
    EXTENDED_VARIANTS["it23"], fuse_libm_guard=True
)
EXTENDED_VARIANTS["it25"] = dict(
    EXTENDED_VARIANTS["it24"], keep_libm_tables=True
)
EXTENDED_VARIANTS["it26"] = dict(EXTENDED_VARIANTS["it25"], table_layout=True)
EXTENDED_VARIANTS["it27"] = dict(EXTENDED_VARIANTS["it26"], lowrank_hess=True)
EXTENDED_VARIANTS["it28"] = dict(
    EXTENDED_VARIANTS["it27"], drop_pair_tables=True
)
EXTENDED_VARIANTS["it29"] = dict(
    EXTENDED_VARIANTS["it28"], share_elementwise=True
)
EXTENDED_VARIANTS["it30"] = dict(
    EXTENDED_VARIANTS["it29"], scalar_chain="cost"
)
EXTENDED_VARIANTS["it31"] = dict(EXTENDED_VARIANTS["it30"], split_affine=True)
# Third round: the scalar_chain path chosen from estimates (generation time).
EXTENDED_VARIANTS["it32"] = dict(
    EXTENDED_VARIANTS["it31"], scalar_chain="estimate"
)
EXTENDED_VARIANTS["it33"] = dict(EXTENDED_VARIANTS["it32"], split_acc=2)
for _key, _settings in EXTENDED_VARIANTS.items():
    VARIANTS[_key] = dict(
        key=_key,
        id=f"td_{_key}",
        token=_key,
        label=_key.upper(),
        **_settings,
    )

#: Standard names (NAMING.md): one cumulative chain of *stages* e0..e8 from
#: the most basic generator to the chain rule, the *steps* p01..p22 of the
#: iterative optimisation (each adds one option to the previous step, except
#: p07, which drops ``sep_sums`` again, see the iteration log), and ablation
#: variants named after their base stage.  The option sets are exactly those
#: of the legacy keys (a*, c*, it*), which are kept for the recorded kernels.
LEGACY_KEYS = {
    # main chain
    "e0": "a0",
    "e1": "a2",
    "e2": "a3",
    "e3": "a6",
    "e4": "a10",
    "e5": "a10h",
    "e6": "c6",
    "e7": "c6n",
    "e8": "c6m",
    # ablation variants
    "e0c": "a8",
    "e1a": "a1",
    "e2y": "a7",
    "e2k": "a4",
    "e2p": "a5",
    "e2a": "a9",
    "e2h": "a3h",
    "e3h": "a6h",
    "e6c": "c3",
    "e6l": "c6l",
    "e6f": "c6f",
    "e7y": "c6s",
    "e7x": "c6x",
    "e7r": "c6r",
    # iterative optimisation
    **{
        f"p{k:02d}": f"it{k:02d}"
        for k in range(1, 100)
        if f"it{k:02d}" in EXTENDED_VARIANTS
    },
}
#: ``e2w`` (single W6 kernel with optional outputs) is generated by
#: bench2/scripts/gen_td.py from ``e2`` and is not a setting of to_fortran.
STAGE_VARIANTS = {
    key: dict(
        key=key,
        id=f"td-{key}",
        token=key,
        label=key,
        legacy=old,
        **EXTENDED_VARIANTS[old],
    )
    for key, old in LEGACY_KEYS.items()
}

#: Workload id -> ``derivatives=`` string (frozen interface contract).
WORKLOADS = {
    "w0": "none",
    "w1": "dV",
    "w2": "dT",
    "w3": "dn",
    "w4": "dT,dV,dn",
    "w5a": "dV,dV2",
    "w5b": "dT,dV,dT2,dTdV",
    "w5c": "dn,dndn",
    "w5d": "dn,dndV,dndT",
    "w6": "dT,dV,dn,dT2,dV2,dndn,dTdV,dndT,dndV",
    # excess Gibbs models, variables (n, T)
    "g0": "none",
    "g1": "dn",
    "g2": "dT,dT2",
    "g3": "dn,dndn",
    "g4": "dT,dT2,dn,dndT,dndn",
}

#: Indentation used by the emitted bare subroutine.
_INDENT = "   "


def variant_settings(variant):
    """Return the settings dict of a generation variant.

    Parameters
    ----------
    variant : str
        ``"v1"``, ``"v2"``, ``"v3"``, or a protocol method id such as
        ``"td_v2_partial"``.

    Returns
    -------
    dict
        The matching entry of :data:`VARIANTS`.

    Raises
    ------
    ValueError
        If the variant is unknown.
    """
    key = str(variant).strip().lower()
    if key in VARIANTS:
        return VARIANTS[key]
    if key in STAGE_VARIANTS:
        return STAGE_VARIANTS[key]
    for entry in list(VARIANTS.values()) + list(STAGE_VARIANTS.values()):
        if key == entry["id"]:
            return entry
    raise ValueError(
        f"unknown variant {variant!r}; expected one of "
        + ", ".join(sorted(VARIANTS))
    )


def _normalise_workload(workload):
    """Return the canonical workload id (lower case, e.g. ``w5a``)."""
    key = str(workload).strip().lower()
    if key.startswith("w") and key[1:] in WORKLOADS:
        key = "w" + key[1:]
    if key not in WORKLOADS:
        raise ValueError(
            f"unknown workload {workload!r}; expected one of "
            + ", ".join(WORKLOADS)
        )
    return key


def _splice_uses(code, uses):
    """Insert the model ``use`` lines before ``implicit none``.

    The splice is deterministic and happens at generation time, never at
    build time: the generated file is the final artifact.

    Parameters
    ----------
    code : str
        Emitted bare subroutine source.
    uses : list of str
        ``use`` clauses as written in the ``.model`` file.

    Returns
    -------
    str
        Source with the ``use`` lines spliced in.
    """
    if not uses:
        return code
    lines = code.splitlines()
    for position, line in enumerate(lines):
        if line.strip() == "implicit none":
            lines[position:position] = [
                f"{_INDENT}use {use_text}" for use_text in uses
            ]
            break
    else:  # pragma: no cover - to_fortran always emits implicit none
        raise RuntimeError("generated code has no 'implicit none' line")
    return "\n".join(lines) + "\n"


def _normalised_options(settings, optional_derivatives):
    """Build the validated options of a variant, silencing the warning.

    The benchmark variants deliberately enable options that some
    combinations ignore, so the ``UserWarning`` of
    :meth:`CodegenOptions.validate` is expected noise here; the ignored
    names are returned instead so that they are still recorded.

    Parameters
    ----------
    settings : dict
        Variant settings as returned by :func:`variant_settings`.
    optional_derivatives : bool
        Declare every output as ``optional, intent(out)``.

    Returns
    -------
    tuple
        The normalised :class:`CodegenOptions` and the tuple of the names
        of the options that were ignored and disabled.
    """
    codegen_settings = {
        key: settings[key]
        for key in ("array_cse", "share", *_EXTRA_KEYS)
        if key in settings
    }
    options = CodegenOptions.from_legacy(
        **codegen_settings, optional_derivatives=optional_derivatives
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return options.validate()


def generate_workload(
    model_path,
    workload,
    variant,
    out_dir,
    optional_derivatives=True,
    subroutine_name=None,
    write=True,
):
    """Generate one workload of one variant and measure the generation.

    Parameters
    ----------
    model_path : str
        Path to the ``.model`` file.
    workload : str
        Workload id (``w0`` ... ``w6``, ``w5a``, ``w5b``, ``w5c``,
        ``w5d``), case insensitive.
    variant : str
        ``v1``/``v2``/``v3`` or a protocol method id (see
        :func:`variant_settings`).
    out_dir : str
        Directory the generated ``.f90`` file is written into.
    optional_derivatives : bool, optional
        Declare every output as ``optional, intent(out)`` (D-014 default).
    subroutine_name : str, optional
        Name of the emitted subroutine.  Defaults to
        ``<model>_<variant key>_<workload>``, e.g. ``pr76_v1_w0``.
    write : bool, optional
        Write the ``.f90`` file (default).  With ``False`` the metadata is
        still returned, with ``path`` pointing at where the file would go.

    Returns
    -------
    dict
        Metadata with keys ``model``, ``variant``, ``variant_id``,
        ``variant_token``, ``workload``, ``derivatives``,
        ``subroutine_name``, ``optional_derivatives``, ``array_cse``,
        ``share``, ``spliced_uses``, ``path``, ``generation_time_s``,
        ``parse_time_s``, ``diff_time_s``, ``emit_time_s``, ``lines``,
        ``bytes``, ``operation_count``, ``operation_count_tool``,
        ``n_temporaries``, ``ignored_options`` (names of the options the
        combination ignored) and ``source`` (the emitted file content).
    """
    workload = _normalise_workload(workload)
    settings = variant_settings(variant)

    with open(model_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    started = time.perf_counter()
    phase = time.perf_counter()
    raw = read_sections(text, model_path)
    ir = build_ir(raw, model_path)
    parse_time_s = time.perf_counter() - phase

    phase = time.perf_counter()
    diff, _selection = build_diff(ir, derivatives=WORKLOADS[workload])
    diff_time_s = time.perf_counter() - phase

    name = os.path.splitext(os.path.basename(str(model_path)))[0]
    model_name = ir.model_name or name
    routine = subroutine_name or f"{model_name}_{settings['key']}_{workload}"

    options, ignored_options = _normalised_options(
        settings, optional_derivatives
    )
    stats = {"ignored_options": ignored_options}
    phase = time.perf_counter()
    code = to_fortran(
        diff,
        derivatives=WORKLOADS[workload],
        name="f",
        subroutine_name=routine,
        integer_args=ir.dimensions,
        module_symbols=ir.imported,
        options=options,
        stats=stats,
        symmetric_params=getattr(ir, "symmetric", ()),
    )
    emit_time_s = time.perf_counter() - phase

    code = _splice_uses(code, ir.uses)
    generation_time_s = time.perf_counter() - started

    path = os.path.join(out_dir, f"{routine}.f90")
    if write:
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(code)

    lines = code.count("\n")
    return {
        "model": model_name,
        "variant": settings["key"],
        "variant_id": settings["id"],
        "variant_token": settings["token"],
        "workload": workload,
        "derivatives": WORKLOADS[workload],
        "subroutine_name": routine,
        "optional_derivatives": bool(optional_derivatives),
        "array_cse": settings["array_cse"],
        "share": settings["share"],
        **{key: settings[key] for key in _EXTRA_KEYS if key in settings},
        "ignored_options": ignored_options,
        "spliced_uses": list(ir.uses),
        "path": path,
        "generation_time_s": generation_time_s,
        "parse_time_s": parse_time_s,
        "diff_time_s": diff_time_s,
        "emit_time_s": emit_time_s,
        "lines": lines,
        "bytes": len(code.encode("utf-8")),
        "operation_count": stats.get("operation_count"),
        "operation_count_tool": stats.get("operation_count_tool"),
        "scalar_chain_used": stats.get("scalar_chain_used"),
        "scalar_chain_estimate": stats.get("scalar_chain_estimate"),
        "scalar_operation_count": stats.get("scalar_operation_count"),
        "n_temporaries": stats.get("n_temporaries"),
        "source": code,
    }
