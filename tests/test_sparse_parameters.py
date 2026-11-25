"""``sparse_parameters``: work behind ``if (v(i, k) /= 0)``.

The group-count matrix of UNIFAC is mostly zero.  The option guards the
terms that carry an element of it and the tables read only through such
terms.  Kernels with and without the option are compiled and compared; the
structural tests pin where the guards go and where they must not.
"""

import pathlib
import re
import tempfile
import warnings

from fortran_tools import (
    DEBUG_FREE_FORM_FLAGS,
    GFORTRAN,
    compile_and_run,
)

from model_harness import parse

import numpy as np

import pytest

import sympy as sp

from thermodiff import DiffPlz, T, idx_function, n, sum_components
from thermodiff.codegen.options import CodegenOptions
from thermodiff.codegen.sparse import split_by_factor
from thermodiff.core.easy_sums import sum_custom
from thermodiff.fortran import to_fortran

REL_TOLERANCE = 1e-13
SPARSE = ("v",)
FIRST_DERIVATIVES = "dT,dT2,dn,dndT,dndn"
# Input name -> shape as a tuple of extent names.
UNIFAC_SHAPES = {
    "Qk": ("ng",),
    "n": ("nc",),
    "qs": ("nc",),
    "rs": ("nc",),
    "th": ("nc", "ng"),
    "v": ("nc", "ng"),
    "psi": ("ng", "ng"),
    "dpsidT": ("ng", "ng"),
    "dpsidTdT": ("ng", "ng"),
}
OUTPUT_PREFIXES = ("ge", "dge", "f", "df")
EXTENT_NAMES = ("nc", "ng")
QUIET_NAN = "transfer(-2251799813685248_8, 1.0d0)"
KERNEL_NAMES = ("kernel_plain", "kernel_sparse")


def _unifac() -> DiffPlz:
    """Return the UNIFAC excess Gibbs energy (copy of the yaeos spec)."""
    gas_r, z, d = sp.symbols("R z d")
    ng = sp.Symbol("ng", integer=True, positive=True)
    l, s = sp.Idx("l"), sp.Idx("s")
    m, p, k = sp.Idx("m"), sp.Idx("p"), sp.Idx("k")
    qs, rs = sp.IndexedBase("qs"), sp.IndexedBase("rs")
    v, th = sp.IndexedBase("v"), sp.IndexedBase("th")
    group_q = sp.IndexedBase("Qk")
    psi_mk = idx_function("psi")(m, k, T)
    ntot = sum_components(n[s], s)
    comb = (
        sum_components(n[l] * d * sp.log(rs[l]), l)
        - ntot * sp.log(sum_components(n[s] * rs[s] ** d, s))
        + ntot * sp.log(ntot)
        + z
        / 2
        * sum_components(
            n[l]
            * qs[l]
            * (
                sp.log(qs[l] / rs[l])
                - sp.log(sum_components(n[s] * qs[s], s))
                + sp.log(sum_components(n[s] * rs[s], s))
            ),
            l,
        )
    )
    moles_m = sum_components(n[s] * v[s, m], s)
    moles_q = sum_custom(
        sum_components(n[s] * v[s, p], s) * group_q[p], p, 1, ng
    )
    pure = sum_components(
        n[l] * v[l, k] * sp.log(sum_custom(th[l, m] * psi_mk, m, 1, ng)), l
    )
    mix = sum_components(n[s] * v[s, k], s) * sp.log(
        sum_custom(moles_m * group_q[m] / moles_q * psi_mk, m, 1, ng)
    )
    res = sum_custom(group_q[k] * (pure - mix), k, 1, ng)
    return DiffPlz(
        gas_r * T * (comb + res),
        internal_functions=[psi_mk],
        indexes=[l, s, m, p, k],
        name="ge",
    )


def _weighted_logs() -> DiffPlz:
    """Return a model with the sparse factor inside a sum and a square.

    ``f = sum_k (sum_l n_l w_lk log(1 + sum_m c_lm psi_mk(T)))**2 / N``: the
    mole numbers appear in the guarded terms, so ``dni`` and ``dnidnj``
    differentiate through them.
    """
    l, m, k, s = sp.Idx("l"), sp.Idx("m"), sp.Idx("k"), sp.Idx("s")
    ng = sp.Symbol("ng", integer=True, positive=True)
    w, c = sp.IndexedBase("v"), sp.IndexedBase("th")
    psi_mk = idx_function("psi")(m, k, T)
    inner = sp.log(1 + sum_custom(c[l, m] * psi_mk, m, 1, ng))
    weighted = sum_components(n[l] * w[l, k] * inner, l)
    expr = sum_custom(weighted**2, k, 1, ng) / sum_components(n[s], s)
    return DiffPlz(
        expr,
        internal_functions=[psi_mk],
        indexes=[l, m, k, s],
        name="f",
    )


MODELS = {"unifac": _unifac, "weighted_logs": _weighted_logs}


def _kernel(
    model: str,
    name: str,
    derivatives: str = FIRST_DERIVATIVES,
    **settings,
) -> str:
    """Return the kernel of ``model`` as a subroutine called ``name``."""
    preset = settings.pop("preset", "standard")
    options = CodegenOptions.from_preset(
        preset, optional_derivatives=False, **settings
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return to_fortran(
            MODELS[model](),
            options=options,
            derivatives=derivatives,
            subroutine_name=name,
        )


def _group_counts(rng, nc: int, ng: int, dense: bool) -> np.ndarray:
    """Return a group-count matrix: 1-3 groups per component, or dense."""
    counts = np.zeros((nc, ng))
    if dense:
        return rng.integers(1, 4, size=(nc, ng)).astype(float)
    for component in range(nc):
        groups = rng.choice(ng, size=rng.integers(1, 4), replace=False)
        counts[component, groups] = rng.integers(1, 4, size=len(groups))
    counts[:, 0] += counts.sum(axis=1) == 0
    return counts


def _inputs(nc: int, ng: int, dense: bool, seed: int = 7) -> dict:
    """Return random positive inputs of the two models."""
    rng = np.random.default_rng(seed)
    counts = _group_counts(rng, nc, ng, dense)
    group_q = rng.uniform(0.4, 2.0, ng)
    weights = counts * group_q
    return {
        "nc": nc,
        "ng": ng,
        "T": 313.0,
        "R": 0.08314,
        "d": 0.75,
        "z": 10.0,
        "Qk": group_q,
        "n": rng.uniform(0.2, 1.5, nc),
        "qs": rng.uniform(0.8, 3.0, nc),
        "rs": rng.uniform(0.8, 3.5, nc),
        "th": weights / weights.sum(axis=1, keepdims=True),
        "v": counts,
        "psi": rng.uniform(0.5, 1.5, (ng, ng)),
        "dpsidT": rng.uniform(-0.01, 0.01, (ng, ng)),
        "dpsidTdT": rng.uniform(-0.001, 0.001, (ng, ng)),
    }


def _extents(name: str, rank: int, is_output: bool) -> str:
    """Return the Fortran extents of an argument of the two models."""
    if rank == 1:
        return "ng" if name == "Qk" else "nc"
    if is_output:
        return "nc,nc"
    return "nc,ng" if name in ("th", "v") else "ng,ng"


def _driver(unguarded: str, guarded: str) -> tuple[str, list[str]]:
    """Return a program that runs both kernels on the data of ``in.txt``.

    The input file holds ``nc`` and ``ng`` and then every other input in
    argument order (arrays in column-major order).

    Returns
    -------
    tuple
        The Fortran source and the names of the inputs, in file order.
    """
    module = f"module kernels\ncontains\n{unguarded}\n{guarded}\nend module\n"
    joined = re.sub(r"&\s*\n\s*&?\s*", "", module)
    _name, args, ranks, _ = parse(joined)
    outputs = [a for a in args if a.startswith(OUTPUT_PREFIXES)]
    inputs = [a for a in args if a not in outputs and a not in EXTENT_NAMES]
    declare = ["integer :: nc, ng"]
    body = ["open(newunit=u, file='in.txt')", "read(u,*) nc, ng"]
    for arg in inputs:
        if ranks[arg] == 0:
            declare.append(f"real(8) :: {arg}")
        else:
            declare.append(
                f"real(8), allocatable :: {arg}({','.join(':' * ranks[arg])})"
            )
            body.append(f"allocate({arg}({_extents(arg, ranks[arg], False)}))")
        body.append(f"read(u,*) {arg}")
    body.append("close(u)")
    calls = ["nc=nc", "ng=ng"] + [f"{a}={a}" for a in inputs]
    for position, kernel in enumerate(KERNEL_NAMES):
        suffix = str(position)
        for out in outputs:
            rank = ranks[out]
            name = f"{out}_{suffix}"
            if rank:
                dims = ",".join(":" * rank)
                declare.append(f"real(8), allocatable :: {name}({dims})")
                body.append(f"allocate({name}({_extents(out, rank, True)}))")
            else:
                declare.append(f"real(8) :: {name}")
        args_out = [f"{o}={o}_{suffix}" for o in outputs]
        body.append(f"call {kernel}({', '.join(calls + args_out)})")
        body.append(f"open(newunit=u, file='out{suffix}.txt')")
        body.extend(f"write(u,'(es30.17e3)') {o}_{suffix}" for o in outputs)
        body.append("close(u)")
    text = (
        module
        + "program drive\nuse kernels\nimplicit none\ninteger :: u\n"
        + "\n".join(declare)
        + "\n"
        + "\n".join(body)
        + "\nend program\n"
    )
    return text, inputs


def _write_inputs(path: pathlib.Path, data: dict, names: list[str]) -> None:
    """Write the input file read by the driver."""
    with path.open("w") as handle:
        handle.write(f"{data['nc']} {data['ng']}\n")
        for name in names:
            values = np.atleast_1d(np.asarray(data[name], dtype=float))
            flat = values.flatten(order="F")
            handle.write(" ".join(f"{value:.17e}" for value in flat) + "\n")


def _poison_arrays(code: str) -> str:
    """Fill every local array of the kernel with a quiet NaN on entry.

    An entry of a guarded table that is read where the guard skipped it
    would then turn the result into NaN, which proves that skipped entries
    are never read.
    """
    lines = code.split("\n")
    local = [
        position
        for position, line in enumerate(lines)
        if line.strip().startswith("real(8) ::") and "intent" not in line
    ]
    names = [
        name
        for position in local
        for name in re.findall(r"\b(t\d+)\(", lines[position])
    ]
    anchor = max(
        position
        for position, line in enumerate(lines)
        if line.strip().startswith("integer ::") and "intent" not in line
    )
    poison = [f"   {name} = {QUIET_NAN}" for name in names]
    return "\n".join(lines[: anchor + 1] + poison + lines[anchor + 1 :])


def _run_pair(
    model: str, data: dict, settings: dict, derivatives: str
) -> tuple[np.ndarray, np.ndarray]:
    """Run the kernel without and with the option; return both outputs."""
    plain = _kernel(model, KERNEL_NAMES[0], derivatives, **settings)
    guarded = _kernel(
        model,
        KERNEL_NAMES[1],
        derivatives,
        sparse_parameters=SPARSE,
        **settings,
    )
    source, names = _driver(plain, _poison_arrays(guarded))
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        (workdir / "drive.f90").write_text(source)
        _write_inputs(workdir / "in.txt", data, names)
        compile_and_run(
            [workdir / "drive.f90"], workdir, DEBUG_FREE_FORM_FLAGS
        )
        return (
            np.loadtxt(workdir / "out0.txt"),
            np.loadtxt(workdir / "out1.txt"),
        )


def _assert_agree(guarded: np.ndarray, plain: np.ndarray) -> None:
    """Compare at ``rel=1e-13`` of each value or of the largest output.

    A term guard splits an accumulation in two, which reassociates it; an
    element that is a difference of larger terms then differs in its last
    bits relative to itself, but not relative to the largest output.
    """
    scale = np.max(np.abs(plain))
    np.testing.assert_allclose(
        guarded, plain, rtol=REL_TOLERANCE, atol=REL_TOLERANCE * scale
    )


CASES = [(3, 6), (5, 8), (10, 12)]
UNIFAC_SETS = {
    "value": "none",
    "gradient": "dn",
    "full": FIRST_DERIVATIVES,
}


@pytest.mark.skipif(GFORTRAN is None, reason="gfortran not available")
class TestNumerics:
    """Guarded and unguarded kernels agree."""

    @pytest.mark.slow
    @pytest.mark.parametrize("dense", [False, True])
    @pytest.mark.parametrize("nc, ng", CASES)
    @pytest.mark.parametrize("preset", ["standard", "fastest"])
    def test_unifac(self, preset, nc, ng, dense):
        data = _inputs(nc, ng, dense)
        plain, guarded = _run_pair(
            "unifac", data, {"preset": preset}, FIRST_DERIVATIVES
        )
        assert np.all(np.isfinite(guarded))
        _assert_agree(guarded, plain)

    @pytest.mark.slow
    @pytest.mark.parametrize("derivatives", ["dn", "dT,dT2"])
    def test_unifac_other_output_sets(self, derivatives):
        data = _inputs(6, 9, dense=False)
        plain, guarded = _run_pair(
            "unifac", data, {"preset": "fastest"}, derivatives
        )
        _assert_agree(guarded, plain)

    @pytest.mark.slow
    @pytest.mark.parametrize("preset", ["standard", "fastest", "minimal"])
    @pytest.mark.parametrize("dense", [False, True])
    def test_sparse_factor_in_sums_and_derivatives(self, preset, dense):
        data = _inputs(7, 9, dense)
        plain, guarded = _run_pair(
            "weighted_logs",
            data,
            {"preset": preset},
            "dT,dT2,dn,dndT,dndn",
        )
        _assert_agree(guarded, plain)


def _guard_conditions(code: str) -> list[str]:
    """Return the zero tests of the kernel, in order."""
    return re.findall(r"if \((v\([^)]*\)) /= 0\.0d0\) then", code)


def _cheap_model() -> DiffPlz:
    """Return a model whose sparse terms hold no costly operation."""
    l, k = sp.Idx("l"), sp.Idx("k")
    ng = sp.Symbol("ng", integer=True, positive=True)
    w, weight = sp.IndexedBase("v"), sp.IndexedBase("w")
    inner = sum_components(n[l] * w[l, k], l)
    expr = sum_custom(weight[k] * inner, k, 1, ng) * T
    return DiffPlz(expr, indexes=[l, k], name="f")


class TestStructure:
    """Where the guards go, and where they do not."""

    def test_default_has_no_guard(self):
        code = _kernel("unifac", "k", preset="fastest")
        assert "/= 0.0d0" not in code

    @pytest.mark.parametrize("preset", ["standard", "fastest"])
    def test_tables_and_reductions_are_guarded(self, preset):
        code = _kernel("unifac", "k", preset=preset, sparse_parameters=SPARSE)
        assert set(_guard_conditions(code)) == {"v(l, k)", "v(i, k)"}
        table = re.search(
            r"if \(v\(l, k\) /= 0\.0d0\) then\n\s+(t\d+)\(l, k\) = 0", code
        )
        assert table, "the sum table over (l, k) is not guarded"

    def test_cheap_work_is_not_guarded(self):
        options = CodegenOptions.from_preset(
            "standard", sparse_parameters=SPARSE, optional_derivatives=False
        )
        code = to_fortran(_cheap_model(), options=options, derivatives="dn")
        assert "/= 0.0d0" not in code

    def test_without_sparse_parameter_in_the_model_nothing_changes(self):
        plain = _kernel("weighted_logs", "k")
        other = CodegenOptions.from_preset(
            "standard", optional_derivatives=False, sparse_parameters=("Qk",)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            code = to_fortran(
                _weighted_logs(),
                options=other,
                derivatives=FIRST_DERIVATIVES,
                subroutine_name="k",
            )
        assert code == plain

    def test_unknown_name_warns_and_is_dropped(self):
        options = CodegenOptions.from_preset(
            "standard",
            optional_derivatives=False,
            sparse_parameters=("nosuch",),
        )
        with pytest.warns(UserWarning, match="nosuch"):
            code = to_fortran(
                _unifac(),
                options=options,
                derivatives="dn",
                subroutine_name="k",
            )
        assert code == _kernel("unifac", "k", "dn")

    def test_option_is_coerced_to_a_tuple_of_names(self):
        assert CodegenOptions(sparse_parameters="v").sparse_parameters == (
            "v",
        )
        options = CodegenOptions(sparse_parameters=["v", "w", "v"])
        assert options.sparse_parameters == ("v", "w")

    def test_inert_presets_keep_it_empty(self):
        for preset in ("minimal", "standard", "fastest"):
            assert not CodegenOptions.from_preset(preset).sparse_parameters


class TestSplitByFactor:
    """Terms with the factor go to the guarded part."""

    ELEMENT = sp.IndexedBase("v")[sp.Idx("i"), sp.Idx("k")]
    X, Y = sp.symbols("x y")

    def test_distributes_over_a_sum_holding_the_factor(self):
        expr = self.X * (self.Y + self.ELEMENT * self.Y**2 + 3)
        rest, guarded = split_by_factor(expr, self.ELEMENT)
        assert sp.expand(rest) == sp.expand(self.X * (self.Y + 3))
        assert sp.expand(guarded) == sp.expand(
            self.X * self.ELEMENT * self.Y**2
        )

    def test_the_parts_add_up(self):
        expr = self.X * self.ELEMENT + self.Y * (self.X + self.ELEMENT)
        rest, guarded = split_by_factor(expr, self.ELEMENT)
        assert sp.expand(rest + guarded - expr) == 0

    def test_the_factor_in_a_function_stays_in_the_rest(self):
        expr = sp.log(self.ELEMENT) * self.X
        rest, guarded = split_by_factor(expr, self.ELEMENT)
        assert (rest, guarded) == (expr, 0)

    def test_the_factor_in_a_denominator_stays_in_the_rest(self):
        expr = self.X / self.ELEMENT
        assert split_by_factor(expr, self.ELEMENT) == (expr, 0)
