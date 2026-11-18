"""Declarative table of golden cases.

A case is a ``(model, variant, workload)`` triple.  The table below is the
single source of truth for the tiers used by ``golden.py`` and
``check_golden.py``:

``fast``
    The ``examples/`` models plus one medium external model
    (``SRK_MODEL``) restricted to :data:`FAST_EXTERNAL_WORKLOADS`.
``full``
    Every case except those listed in :data:`SLOW_CASES`.
``complete``
    Every case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Variants of the golden matrix (missing ones are skipped with a warning).
MATRIX_VARIANTS: tuple[str, ...] = (
    "e0", "e2", "e3", "e8", "p22", "p32", "p33",
)  # fmt: skip

#: Workloads per model family.
FAMILY_WORKLOADS: dict[str, tuple[str, ...]] = {
    "helmholtz": (
        "w0", "w1", "w2", "w3", "w4", "w5a", "w5b", "w5c", "w5d", "w6",
    ),  # fmt: skip
    "gibbs": ("g0", "g1", "g2", "g3", "g4"),
    # Models whose only variable is n (no T, no V).
    "mole_only": ("g0", "g1", "g3"),
}

#: Directory prefix of the models whose golden is versioned in the repo.
EXAMPLES_DIR = "examples"
#: Medium external model included in the fast tier.
SRK_MODEL = "tests/data/models/srk/srk.model"
FAST_EXTERNAL_WORKLOADS: tuple[str, ...] = ("w0", "w3", "w6")

TIERS: tuple[str, ...] = ("fast", "full", "complete")


class ModelSpec(NamedTuple):
    """A model of the golden table.

    Attributes
    ----------
    path : str
        Model path relative to the repository root.
    family : str
        Key of :data:`FAMILY_WORKLOADS`.
    """

    path: str
    family: str

    @property
    def name(self) -> str:
        """Return the model name (file stem)."""
        return Path(self.path).stem

    @property
    def is_example(self) -> bool:
        """Return whether the golden of this model is versioned."""
        return self.path.startswith(EXAMPLES_DIR + "/")


class Case(NamedTuple):
    """One generation case."""

    model: ModelSpec
    variant: str
    workload: str

    @property
    def relpath(self) -> str:
        """Return the golden file path relative to the golden root."""
        name = self.model.name
        stem = f"{name}_{self.variant}_{self.workload}.f90"
        return f"{name}/{self.variant}/{stem}"


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("examples/molefrac.model", "mole_only"),
    ModelSpec("examples/pr.model", "helmholtz"),
    ModelSpec("tests/data/models/cmr/cmr.model", "helmholtz"),
    ModelSpec("tests/data/models/nrtl/nrtl.model", "gibbs"),
    ModelSpec("tests/data/models/pcsaft/pcsaft.model", "helmholtz"),
    ModelSpec("tests/data/models/pr76/pr76.model", "helmholtz"),
    ModelSpec("tests/data/models/pr76/pr76_sym.model", "helmholtz"),
    ModelSpec("tests/data/models/rkpr/rkpr.model", "helmholtz"),
    ModelSpec("tests/data/models/srk/srk.model", "helmholtz"),
    ModelSpec("tests/data/models/uniquac/uniquac.model", "gibbs"),
)

#: Cases excluded from the ``full`` tier (the coordinator fills this in).
SLOW_CASES: frozenset[Case] = frozenset()


def _all_cases(variants: tuple[str, ...]) -> list[Case]:
    """Return every case of the table for the given variants."""
    return [
        Case(model, variant, workload)
        for model in MODELS
        for variant in variants
        for workload in FAMILY_WORKLOADS[model.family]
    ]


def available_variants() -> tuple[str, ...]:
    """Return the matrix variants known to the generator.

    Missing variants are skipped and a warning is printed.

    Returns
    -------
    tuple of str
        Matrix variants present in ``STAGE_VARIANTS``.
    """
    from thermodiff.bench_hooks import STAGE_VARIANTS

    present = tuple(v for v in MATRIX_VARIANTS if v in STAGE_VARIANTS)
    for missing in sorted(set(MATRIX_VARIANTS) - set(present)):
        print(f"warning: variant {missing!r} not available, skipped")
    return present


def _in_fast_tier(case: Case) -> bool:
    """Return whether a case belongs to the fast tier."""
    if case.model.is_example:
        return True
    return (
        case.model.path == SRK_MODEL
        and case.workload in FAST_EXTERNAL_WORKLOADS
    )


def cases_for_tier(
    tier: str, variants: tuple[str, ...] | None = None
) -> list[Case]:
    """Return the cases of a tier in a deterministic order.

    Parameters
    ----------
    tier : str
        One of :data:`TIERS`.
    variants : tuple of str, optional
        Variants to use; defaults to :func:`available_variants`.

    Returns
    -------
    list of Case
        Cases sorted by model, variant and workload table order.

    Raises
    ------
    ValueError
        If the tier is unknown.
    """
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
    cases = _all_cases(variants or available_variants())
    if tier == "fast":
        return [case for case in cases if _in_fast_tier(case)]
    if tier == "full":
        return [case for case in cases if case not in SLOW_CASES]
    return cases


@dataclass(frozen=True)
class GoldenRoots:
    """Directories where the golden files live.

    Attributes
    ----------
    examples : pathlib.Path
        Root of the versioned golden (``tests/golden``).
    external : pathlib.Path
        Root of the external golden (``$THERMODIFF_GOLDEN_EXT``).
    """

    examples: Path
    external: Path

    def for_model(self, model: ModelSpec) -> Path:
        """Return the golden root that holds a model."""
        return self.examples if model.is_example else self.external

    @classmethod
    def default(cls, repo: Path = REPO_ROOT) -> "GoldenRoots":
        """Return the default roots (honours ``THERMODIFF_GOLDEN_EXT``)."""
        import os

        external = os.environ.get("THERMODIFF_GOLDEN_EXT")
        return cls(
            repo / "tests" / "golden",
            Path(external) if external else repo.parent / "golden_ext",
        )
