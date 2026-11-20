"""Declarative registry of the benchmark generation variants.

Every variant is one :func:`stage` entry of :data:`STAGES`: the key of its
parent stage and only the settings it adds or overrides.  The full settings
of a stage are obtained by walking the parents (:func:`resolve_settings`).
Nothing here is mutated after import; :mod:`benchmarks.td_bench.hooks`
builds its public tables once from this registry.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

__all__ = [
    "EXTRA_KEYS",
    "LEGACY_ALIASES",
    "PROTOCOL_VARIANTS",
    "STAGES",
    "Stage",
    "resolve_settings",
    "stage",
]


@dataclass(frozen=True)
class Stage:
    """One variant: its parent and the settings it adds or overrides.

    Parameters
    ----------
    key : str
        Short name of the stage (``"a3"``, ``"c6m"``, ``"it12"``).
    base : str or None
        Key of the parent stage, or ``None`` for a root stage.
    delta : tuple of (str, object)
        Settings added or overridden relative to the parent, in order.
    """

    key: str
    base: Optional[str]
    delta: Tuple[Tuple[str, Any], ...]


def stage(key: str, base: Optional[str] = None, **settings: Any) -> Stage:
    """Build a :class:`Stage` from keyword settings.

    Parameters
    ----------
    key : str
        Short name of the stage.
    base : str, optional
        Key of the parent stage.
    **settings
        Settings added or overridden relative to the parent.

    Returns
    -------
    Stage
        The immutable registry entry.
    """
    return Stage(key, base, tuple(settings.items()))


#: Optional codegen keys a variant may set (see ``to_fortran``).
EXTRA_KEYS: Tuple[str, ...] = (
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
#: Each one names its parent and only the settings it changes.
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
#: a3p/a6p/a10p pad the rank-2 temporaries (N >= 45 bimodality); a*h hoist
#: loop-invariants; c* differentiate with the chain rule over summation
#: nodes; it01..it33 are the iterative optimisation steps (it23 onwards:
#: the second and third rounds).
STAGES: Tuple[Stage, ...] = (
    stage("a0", base=None, array_cse=False, share=False),
    stage("a1", base=None, array_cse=False, share=True, scalar_cse=False),
    stage("a2", base=None, array_cse=False, share=True),
    stage("a3", base=None, array_cse=True, share=True),
    stage("a4", base="a3", hessian_symmetry=True),
    stage("a5", base="a3", param_symmetry=True),
    stage("a6", base="a5", hessian_symmetry=True),
    stage("a7", base="a3", cse_backend="sympy"),
    stage("a8", base="a0", scalar_cse=True),
    stage("a9", base="a3", sum_alpha=True),
    stage("a10", base="a6", sum_alpha=True),
    stage("a3p", base="a3", array_pad=8),
    stage("a6p", base="a6", array_pad=8),
    stage("a10p", base="a10", array_pad=8),
    stage("a3h", base="a3", hoist_invariants=True),
    stage("a6h", base="a6", hoist_invariants=True),
    stage("a10h", base="a10", hoist_invariants=True),
    stage("c3", base="a3h", chain_rule=True),
    stage("c6", base="a10h", chain_rule=True),
    stage("c6l", base="c6", hoist_loops=True),
    stage("c6f", base="c6l", sum_free_alpha=True, fuse_temps=True),
    stage("c6n", base="c6f", nest_sums=True),
    stage("c6s", base="c6n", cse_backend="sympy"),
    stage("c6x", base="c6n", split_sums=True),
    stage("c6r", base="c6x", split_sqrt=True, nest_shared=True),
    stage("c6m", base="c6r", sym_matvec=True),
    stage("it01", base="c6m", col_major=True),
    stage("it02", base="it01", pair_cse=True),
    stage("it03", base="it02", pair_rates=True),
    stage("it04", base="it03", sum_transpose=True),
    stage("it05", base="it04", sep_sums=True),
    stage("it06", base="it05", sym_rank3=True),
    stage("it07", base="it04", sym_rank3=True),
    stage("it08", base="it04", sep_sums="cost", sym_rank3=True),
    stage("it09", base="it08", nest_general=True),
    stage("it10", base="it09", sym_sum2=True),
    stage("it11", base="it10", cse_expensive=True),
    stage("it12", base="it11", fuse_loops=True),
    stage("it13", base="it12", nested_nodes=True),
    stage("it14", base="it13", sym_perm=True),
    stage("it15", base="it14", fuse_guard=True),
    stage("it16", base="it15", nest_cost=True),
    stage("it17", base="it16", comp_fuse=True),
    stage("it18", base="it17", fuse_outer=True),
    stage("it19", base="it18", diag_table=True),
    stage("it20", base="it19", sib_fuse=True),
    stage("it21", base="it20", dummy_canon=True),
    stage("it22", base="it21", float_canon=True),
    stage("it23", base="it22", contiguous_args=True),
    stage("it24", base="it23", fuse_libm_guard=True),
    stage("it25", base="it24", keep_libm_tables=True),
    stage("it26", base="it25", table_layout=True),
    stage("it27", base="it26", lowrank_hess=True),
    stage("it28", base="it27", drop_pair_tables=True),
    stage("it29", base="it28", share_elementwise=True),
    stage("it30", base="it29", scalar_chain="cost"),
    stage("it31", base="it30", split_affine=True),
    stage("it32", base="it31", scalar_chain="estimate"),
    stage("it33", base="it32", split_acc=2),
)

#: Benchmark protocol variants (D-007): key, method id, directory token,
#: label and the stage whose settings they share.
PROTOCOL_VARIANTS: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("v1", "td_v1_none", "none", "V1 none", "a0"),
    ("v2", "td_v2_partial", "partial", "V2 partial", "a2"),
    ("v3", "td_v3_full", "full", "V3 full", "a3"),
)

#: Standard names (NAMING.md) -> legacy stage keys: the stages e0..e8, their
#: ablation variants and the steps p01.. of the iterative optimisation.
_STAGE_ALIASES = {
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
}
#: ``e2w`` (single W6 kernel with optional outputs) is generated by
#: bench2/scripts/gen_td.py from ``e2`` and is not a setting of to_fortran.
LEGACY_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        **_STAGE_ALIASES,
        # iterative optimisation: p<NN> is it<NN>
        **{
            f"p{entry.key[2:]}": entry.key
            for entry in STAGES
            if entry.key.startswith("it")
        },
    }
)


def resolve_settings(key: str, stages: Mapping[str, Stage]) -> dict:
    """Return the full settings of a stage by walking its parents.

    Parameters
    ----------
    key : str
        Key of the stage to resolve.
    stages : mapping of str to Stage
        Registry indexed by stage key.

    Returns
    -------
    dict
        New dict with the inherited settings first (in parent order) and
        the stage's own delta applied last.
    """
    entry = stages[key]
    inherited = (
        resolve_settings(entry.base, stages) if entry.base is not None else {}
    )
    return {**inherited, **dict(entry.delta)}
