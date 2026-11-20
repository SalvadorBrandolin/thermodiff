"""Benchmark and golden generation hooks (not part of the package).

Re-exports the public names of :mod:`benchmarks.td_bench.hooks`.
"""

from benchmarks.td_bench.hooks import (
    EXTENDED_VARIANTS,
    LEGACY_KEYS,
    STAGE_VARIANTS,
    VARIANTS,
    WORKLOADS,
    generate_workload,
    variant_settings,
)

__all__ = [
    "EXTENDED_VARIANTS",
    "LEGACY_KEYS",
    "STAGE_VARIANTS",
    "VARIANTS",
    "WORKLOADS",
    "generate_workload",
    "variant_settings",
]
