"""Tests for the benchmark hooks and the real PR76 generated files.

The 30 (workload x variant) files are generated once by the session
fixture of ``conftest.py`` and checked against the frozen interface
contract (H-20261028-symbolic-tool-engineer-interface-contract): routine
names, argument order, output ranks and V1/V2/V3 interface identity on
the real files.  Numerical behaviour is checked against central finite
differences of an independent Python evaluation of the PR76 model.
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys

import pytest

from thermodiff.bench_hooks import (
    VARIANTS,
    WORKLOADS,
    generate_workload,
    variant_settings,
)

from conftest import PR76_MODEL, SESSION_ROOT, VARIANT_ORDER, WORKLOAD_ORDER

R_GAS = 0.08314462618  # yaeos/src/constants.f90:7-8 [bar L /(mol K)]


def _find_gfortran():
    """Return the gfortran executable to use, honoring ``FC``."""
    compiler = os.environ.get("FC")
    if compiler and "gfortran" in os.path.basename(compiler).lower():
        return compiler
    return shutil.which("gfortran")


GFORTRAN = _find_gfortran()

YAEOS_CONSTANTS_STUB = (
    "module yaeos__constants\n"
    "  implicit none\n"
    "  real(8), parameter :: R = 0.08314462618d0\n"
    "end module yaeos__constants\n"
)

#: Frozen argument lists per workload (interface contract table).
_BASE_ARGS = ["nc", "V", "T", "Pc", "Tc", "kij", "lij", "n", "w"]
CONTRACT_ARGS = {
    "w0": _BASE_ARGS + ["f"],
    "w1": _BASE_ARGS + ["f", "dfdV"],
    "w2": _BASE_ARGS + ["f", "dfdT"],
    "w3": _BASE_ARGS + ["f", "dfdn"],
    "w4": _BASE_ARGS + ["f", "dfdT", "dfdV", "dfdn"],
    "w5a": _BASE_ARGS + ["f", "dfdV", "dfdV2"],
    "w5b": _BASE_ARGS + ["f", "dfdT", "dfdV", "dfdT2", "dfdTdV"],
    "w5c": _BASE_ARGS + ["f", "dfdn", "dfdndn"],
    "w5d": _BASE_ARGS + ["f", "dfdn", "dfdndT", "dfdndV"],
    "w6": _BASE_ARGS
    + ["f", "dfdT", "dfdV", "dfdn", "dfdT2", "dfdV2", "dfdndn",
       "dfdTdV", "dfdndT", "dfdndV"],
}

#: Expected output ranks per workload (0 = scalar, 1 = (:), 2 = (:, :)).
EXPECTED_RANKS = {
    "w0": {"f": 0},
    "w1": {"f": 0, "dfdV": 0},
    "w2": {"f": 0, "dfdT": 0},
    "w3": {"f": 0, "dfdn": 1},
    "w4": {"f": 0, "dfdT": 0, "dfdV": 0, "dfdn": 1},
    "w5a": {"f": 0, "dfdV": 0, "dfdV2": 0},
    "w5b": {"f": 0, "dfdT": 0, "dfdV": 0, "dfdT2": 0, "dfdTdV": 0},
    "w5c": {"f": 0, "dfdn": 1, "dfdndn": 2},
    "w5d": {"f": 0, "dfdn": 1, "dfdndT": 1, "dfdndV": 1},
    "w6": {"f": 0, "dfdT": 0, "dfdV": 0, "dfdn": 1, "dfdT2": 0,
           "dfdV2": 0, "dfdndn": 2, "dfdTdV": 0, "dfdndT": 1,
           "dfdndV": 1},
}


def _signature(source):
    """Return (subroutine name, argument list) of the emitted code."""
    joined = []
    for line in source.splitlines():
        joined.append(line.strip().rstrip("&").strip())
        if line.rstrip().endswith(")"):
            break
    statement = " ".join(joined)
    name = statement[len("subroutine ") : statement.index("(")].strip()
    inner = statement[statement.index("(") + 1 : statement.rindex(")")]
    return name, [arg.strip() for arg in inner.split(",")]


_DECL_RE = re.compile(
    r"real\(8\), (?:(optional), )?intent\(out\) :: (\w+)(?:\((.*)\))?$"
)


def _output_ranks(source):
    """Return {output name: rank} from the emitted declarations."""
    ranks = {}
    for line in source.splitlines():
        match = _DECL_RE.match(line.strip())
        if match:
            dims = match.group(3) or ""
            ranks[match.group(2)] = 0 if not dims else dims.count(",") + 1
    return ranks


def _interface(source):
    """Return everything but the local temporaries and the body."""
    name, args = _signature(source)
    declarations = [
        line.strip()
        for line in source.splitlines()
        if "intent(" in line
    ]
    return name, args, declarations


class TestHooksContract:
    """bench_hooks.generate_workload metadata and behaviour."""

    def test_variant_settings(self):
        assert variant_settings("v1") == VARIANTS["v1"]
        assert variant_settings("td_v2_partial") == VARIANTS["v2"]
        assert variant_settings("V3") == VARIANTS["v3"]
        assert VARIANTS["v1"]["array_cse"] is False
        assert VARIANTS["v1"]["share"] is False
        assert VARIANTS["v2"] == dict(VARIANTS["v2"], share=True)
        assert VARIANTS["v3"]["array_cse"] is True

    def test_workload_table_matches_contract(self):
        assert WORKLOADS["w0"] == "none"
        assert WORKLOADS["w3"] == "dn"
        assert WORKLOADS["w5c"] == "dn,dndn"
        assert WORKLOADS["w5d"] == "dn,dndV,dndT"
        assert (
            WORKLOADS["w6"]
            == "dT,dV,dn,dT2,dV2,dndn,dTdV,dndT,dndV"
        )

    def test_metadata_keys_and_metrics(self, tmp_path):
        metadata = generate_workload(
            PR76_MODEL, "w3", "v2", str(tmp_path)
        )
        required = {
            "model", "variant", "variant_id", "variant_token", "workload",
            "derivatives", "subroutine_name", "optional_derivatives",
            "array_cse", "share", "spliced_uses", "path",
            "generation_time_s", "parse_time_s", "diff_time_s",
            "emit_time_s", "lines", "bytes", "operation_count",
            "operation_count_tool", "n_temporaries", "source",
        }
        assert required <= set(metadata)
        assert metadata["optional_derivatives"] is True  # D-014 default
        assert metadata["operation_count_tool"] == "sympy.count_ops(visual=False)"
        assert isinstance(metadata["operation_count"], int)
        assert metadata["generation_time_s"] >= 0.0
        assert metadata["parse_time_s"] >= 0.0
        assert metadata["diff_time_s"] >= 0.0
        assert metadata["emit_time_s"] >= 0.0
        # path is inside the requested out_dir and the metrics describe
        # the file that is written.
        assert os.path.dirname(metadata["path"]) == str(tmp_path)
        assert metadata["path"].startswith(str(tmp_path))
        assert metadata["lines"] == metadata["source"].count("\n")
        assert metadata["bytes"] == len(metadata["source"].encode("utf-8"))
        with open(metadata["path"], "r", encoding="utf-8") as handle:
            assert handle.read() == metadata["source"]

    def test_generate_workload_is_deterministic(self, tmp_path):
        first = generate_workload(PR76_MODEL, "w3", "v2", str(tmp_path))
        second = generate_workload(PR76_MODEL, "w3", "v2", str(tmp_path))
        assert first["source"] == second["source"]
        assert first["operation_count"] == second["operation_count"]
        assert first["n_temporaries"] == second["n_temporaries"]

    def test_unknown_workload_and_variant_raise(self, tmp_path):
        with pytest.raises(ValueError, match="unknown workload"):
            generate_workload(PR76_MODEL, "w9", "v1", str(tmp_path))
        with pytest.raises(ValueError, match="unknown variant"):
            generate_workload(PR76_MODEL, "w0", "v9", str(tmp_path))

    def test_subroutine_name_default(self, tmp_path):
        metadata = generate_workload(PR76_MODEL, "w5a", "v3", str(tmp_path))
        assert metadata["subroutine_name"] == "pr76_v3_w5a"
        assert os.path.basename(metadata["path"]) == "pr76_v3_w5a.f90"


class TestPr76FrozenContract:
    """The real generated files match the frozen interface contract."""

    @pytest.mark.parametrize("variant", VARIANT_ORDER)
    @pytest.mark.parametrize("workload", WORKLOAD_ORDER)
    def test_names_and_arguments(self, pr76_generated, variant, workload):
        metadata = pr76_generated["meta"][(variant, workload)]
        name, args = _signature(metadata["source"])
        assert name == f"pr76_{variant}_{workload}"
        assert args == CONTRACT_ARGS[workload]

    @pytest.mark.parametrize("variant", VARIANT_ORDER)
    @pytest.mark.parametrize("workload", WORKLOAD_ORDER)
    def test_output_ranks(self, pr76_generated, variant, workload):
        metadata = pr76_generated["meta"][(variant, workload)]
        assert _output_ranks(metadata["source"]) == EXPECTED_RANKS[workload]

    @pytest.mark.parametrize("workload", WORKLOAD_ORDER)
    def test_interface_identical_across_variants(
        self, pr76_generated, workload
    ):
        # Identical argument lists and declarations; the only difference
        # is the variant token inside the routine name (contract).
        interfaces = []
        names = []
        for variant in VARIANT_ORDER:
            name, args, declarations = _interface(
                pr76_generated["meta"][(variant, workload)]["source"]
            )
            names.append(name)
            interfaces.append((args, declarations))
        assert interfaces[0] == interfaces[1]
        assert interfaces[1] == interfaces[2]
        assert names == [f"pr76_{v}_{workload}" for v in VARIANT_ORDER]

    @pytest.mark.parametrize("variant", VARIANT_ORDER)
    @pytest.mark.parametrize("workload", WORKLOAD_ORDER)
    def test_outputs_are_optional(self, pr76_generated, variant, workload):
        source = pr76_generated["meta"][(variant, workload)]["source"]
        ranks = _output_ranks(source)
        assert ranks  # outputs exist
        for line in source.splitlines():
            if "intent(out)" in line:
                assert "optional" in line

    @pytest.mark.parametrize("variant", VARIANT_ORDER)
    @pytest.mark.parametrize("workload", WORKLOAD_ORDER)
    def test_interface_is_nc_runtime_argument(
        self, pr76_generated, variant, workload
    ):
        source = pr76_generated["meta"][(variant, workload)]["source"]
        assert "integer, intent(in) :: nc" in source
        assert "real(8), intent(in) :: n(:)" in source
        # No fixed-size declaration anywhere.
        assert not re.search(r":: \w+\(\d+", source)

    @pytest.mark.parametrize("variant", VARIANT_ORDER)
    @pytest.mark.parametrize("workload", WORKLOAD_ORDER)
    def test_use_splice_is_uniform(self, pr76_generated, variant, workload):
        source = pr76_generated["meta"][(variant, workload)]["source"]
        lines = [line.strip() for line in source.splitlines()]
        use_at = lines.index("use yaeos__constants, only: R")
        assert lines[use_at + 1] == "implicit none"
        # Nothing but the (possibly continued) subroutine statement above.
        assert all(
            "intent(" not in line for line in lines[:use_at]
        )
        assert lines[0].startswith("subroutine ")


@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
class TestPr76Compilation:
    """The emitted files compile; the model matches finite differences."""

    def _prepare_stub(self, tmp_path):
        (tmp_path / "stub.f90").write_text(YAEOS_CONSTANTS_STUB)
        result = subprocess.run(
            [GFORTRAN, "-c", "stub.f90"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize("variant", VARIANT_ORDER)
    @pytest.mark.parametrize("workload", WORKLOAD_ORDER)
    def test_files_compile_standalone(self, pr76_generated, tmp_path, variant, workload):
        # Each metric file compiles on its own (protocol 4.4 warning b).
        self._prepare_stub(tmp_path)
        path = pr76_generated["meta"][(variant, workload)]["path"]
        result = subprocess.run(
            [GFORTRAN, "-c", "-ffree-line-length-0", str(path)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr

    def test_include_wrapper_route(self, pr76_generated, tmp_path):
        # Consumer contract R mechanism: the W6 file is included in a
        # wrapper module and R arrives by host association.  The
        # generation-time use splice must be legal in that context too.
        path = pr76_generated["meta"][("v2", "w6")]["path"]
        wrapper = (
            "module pr76_v2_w6_mod\n"
            "   use yaeos__constants, only: R\n"
            "   implicit none\n"
            "contains\n"
            f"   include '{path}'\n"
            "end module pr76_v2_w6_mod\n"
        )
        (tmp_path / "wrapper.f90").write_text(wrapper)
        self._prepare_stub(tmp_path)
        result = subprocess.run(
            [GFORTRAN, "-c", "-ffree-line-length-0", "wrapper.f90"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr

    def _run_w6(self, pr76_generated, tmp_path, variant, state):
        # The generated file is a bare subroutine with assumed-shape
        # arguments: the consumer contract hosts it in a wrapper module
        # and the driver uses that module (explicit interface).
        path = pr76_generated["meta"][(variant, "w6")]["path"]
        wrapper = (
            f"module pr76_{variant}_w6_mod\n"
            "   implicit none\n"
            "contains\n"
            f"   include '{path}'\n"
            f"end module pr76_{variant}_w6_mod\n"
        )
        self._prepare_stub(tmp_path)
        (tmp_path / "wrapper.f90").write_text(wrapper)
        (tmp_path / "driver.f90").write_text(_driver_w6(state, variant))
        exe = tmp_path / "a.out"
        result = subprocess.run(
            [
                GFORTRAN,
                "-O2",
                "-ffree-line-length-0",
                "-o",
                str(exe),
                "stub.f90",
                "wrapper.f90",
                "driver.f90",
            ],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        run = subprocess.run(
            [str(exe)], capture_output=True, text=True, cwd=tmp_path
        )
        assert run.returncode == 0, run.stderr
        values = [float(value) for value in run.stdout.split()]
        assert values  # the driver really printed the outputs
        return values

    @pytest.mark.parametrize("case", ["nc2", "nc4"])
    @pytest.mark.parametrize("variant", VARIANT_ORDER)
    def test_w6_matches_finite_differences(
        self, pr76_generated, tmp_path, variant, case
    ):
        state = STATES[case]
        values = self._run_w6(pr76_generated, tmp_path, variant, state)
        reference = _fd_reference(state)
        assert values == pytest.approx(reference, rel=1e-5, abs=1e-6)

    @pytest.mark.parametrize("case", ["nc2", "nc4"])
    def test_variants_agree_on_pr76(self, pr76_generated, tmp_path, case):
        state = STATES[case]
        results = {
            variant: self._run_w6(pr76_generated, tmp_path, variant, state)
            for variant in VARIANT_ORDER
        }
        assert results["v1"] == pytest.approx(results["v2"], rel=1e-10)
        assert results["v2"] == pytest.approx(results["v3"], rel=1e-10)


class TestDriverScript:
    """scripts/generate-symbolic-variants.py writes files + manifests."""

    def test_driver_writes_files_and_manifests(self, tmp_path):
        script = os.path.join(
            SESSION_ROOT, "scripts", "generate-symbolic-variants.py"
        )
        out_root = tmp_path / "generated"
        result = subprocess.run(
            [
                sys.executable,
                script,
                "--variants",
                "v1",
                "--workloads",
                "w0",
                "w3",
                "--out-root",
                str(out_root),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        variant_dir = out_root / "pr76_v1_none"
        assert (variant_dir / "pr76_v1_w0.f90").exists()
        assert (variant_dir / "pr76_v1_w3.f90").exists()
        manifest = json.loads((variant_dir / "manifest.json").read_text())
        assert manifest["variant"]["id"] == "td_v1_none"
        assert manifest["optional_derivatives"] is True
        entries = {entry["workload"]: entry for entry in manifest["files"]}
        assert set(entries) == {"w0", "w3"}
        for entry in manifest["files"]:
            for key in (
                "generation_time_s",
                "parse_time_s",
                "diff_time_s",
                "emit_time_s",
                "lines",
                "bytes",
                "operation_count",
                "operation_count_tool",
                "n_temporaries",
                "derivatives",
                "subroutine_name",
                "path",
            ):
                assert key in entry
        aggregate = json.loads((out_root / "manifest.json").read_text())
        assert len(aggregate["files"]) == 2

    def test_fresh_process_generations_are_byte_identical(self, tmp_path):
        """Hash-seed variation must not alter emitted Fortran bytes."""
        script = os.path.join(
            SESSION_ROOT, "scripts", "generate-symbolic-variants.py"
        )
        workloads = (
            "w0", "w1", "w2", "w3", "w4", "w5a", "w5b", "w5c",
            "w5d", "w6",
        )
        # Generate every one of the 30 variant/workload combinations in two
        # fresh processes, with distinct seeds. All output stays under tmp_path.
        for variant in VARIANT_ORDER:
            for workload in workloads:
                generated_bytes = []
                for seed in ("11", "29"):
                    out_root = tmp_path / f"seed-{seed}" / variant / workload
                    env = dict(os.environ, PYTHONHASHSEED=seed)
                    result = subprocess.run(
                        [
                            sys.executable, script, "--variants", variant,
                            "--workloads", workload, "--out-root",
                            str(out_root),
                        ],
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                    assert result.returncode == 0, result.stderr
                    files = list(out_root.rglob("*.f90"))
                    assert len(files) == 1
                    generated_bytes.append(files[0].read_bytes())
                assert generated_bytes[0] == generated_bytes[1], (
                    variant, workload
                )


# ---------------------------------------------------------------------------
# Independent PR76 reference evaluation and finite differences
# ---------------------------------------------------------------------------

#: Deterministic state points; kij/lij are symmetric (reshape order safe).
STATES = {
    "nc2": {
        "n": [1.2, 0.8],
        "V": 25.0,
        "T": 320.0,
        "Tc": [310.0, 450.0],
        "Pc": [45.0, 33.0],
        "w": [0.15, 0.25],
        "kij": [[0.0, 0.02], [0.02, 0.0]],
        "lij": [[0.0, 0.01], [0.01, 0.0]],
    },
    "nc4": {
        "n": [1.0, 1.5, 0.7, 1.1],
        "V": 40.0,
        "T": 350.0,
        "Tc": [300.0, 380.0, 420.0, 500.0],
        "Pc": [40.0, 35.0, 30.0, 25.0],
        "w": [0.10, 0.15, 0.20, 0.30],
        "kij": [
            [0.0, 0.01, 0.02, 0.0],
            [0.01, 0.0, 0.015, 0.01],
            [0.02, 0.015, 0.0, 0.005],
            [0.0, 0.01, 0.005, 0.0],
        ],
        "lij": [
            [0.0, 0.005, 0.0, 0.01],
            [0.005, 0.0, 0.005, 0.0],
            [0.0, 0.005, 0.0, 0.005],
            [0.01, 0.0, 0.005, 0.0],
        ],
    },
}


def _pr76_ar(state):
    """Evaluate Ar(n, V, T) of the PR76 model independently.

    Direct Python translation of the yaeos formulas (implementations
    .f90:24-91, alphas.f90:46, mixing_rules/base.f90:35-73, quadratic
    _mixing.f90:64-70, base.f90:79-106); no SymPy involved.
    """
    n = state["n"]
    V = state["V"]
    T = state["T"]
    nc = len(n)
    ntot = sum(n)
    a = []
    b = []
    for p in range(nc):
        kappa = 0.37464 + 1.54226 * state["w"][p] - 0.26993 * state["w"][p] ** 2
        ac = 0.45723553 * R_GAS**2 * state["Tc"][p] ** 2 / state["Pc"][p]
        alpha = (1.0 + kappa * (1.0 - math.sqrt(T / state["Tc"][p]))) ** 2
        a.append(ac * alpha)
        b.append(0.07779607 * R_GAS * state["Tc"][p] / state["Pc"][p])
    bmix = 0.0
    dattract = 0.0
    for p in range(nc):
        for q in range(nc):
            bij = 0.5 * (b[p] + b[q]) * (1.0 - state["lij"][p][q])
            bmix += n[p] * n[q] * bij
            dattract += (
                n[p] * n[q] * math.sqrt(a[p] * a[q]) * (1.0 - state["kij"][p][q])
            )
    bmix = bmix / ntot
    d1 = 1.0 + math.sqrt(2.0)
    d2 = 1.0 - math.sqrt(2.0)
    fterm = math.log((V + d1 * bmix) / (V + d2 * bmix)) / (bmix * (d1 - d2))
    g = R_GAS * math.log(1.0 - bmix / V)
    return -ntot * g * T - dattract * fterm


def _at(state, **changes):
    """Evaluate the reference with the given state changes."""
    new = {key: list(value) if isinstance(value, list) else value
           for key, value in state.items()}
    for key, value in changes.items():
        new[key] = value
    return _pr76_ar(new)


def _shift_n(state, index, delta, **changes):
    n = list(state["n"])
    n[index] += delta
    return _at(state, n=n, **changes)


def _fd_reference(state):
    """Central finite differences of every W6 output, in write order."""
    nc = len(state["n"])
    h1, h2 = 1e-5, 1e-3
    base = _at(state)

    def d_temp():
        return (_at(state, T=state["T"] + h1)
                - _at(state, T=state["T"] - h1)) / (2 * h1)

    def d_vol():
        return (_at(state, V=state["V"] + h1)
                - _at(state, V=state["V"] - h1)) / (2 * h1)

    def d2_temp():
        return (
            _at(state, T=state["T"] + h2)
            - 2 * base
            + _at(state, T=state["T"] - h2)
        ) / h2**2

    def d2_vol():
        return (
            _at(state, V=state["V"] + h2)
            - 2 * base
            + _at(state, V=state["V"] - h2)
        ) / h2**2

    def d2_temp_vol():
        return (
            _at(state, T=state["T"] + h2, V=state["V"] + h2)
            - _at(state, T=state["T"] - h2, V=state["V"] + h2)
            - _at(state, T=state["T"] + h2, V=state["V"] - h2)
            + _at(state, T=state["T"] - h2, V=state["V"] - h2)
        ) / (4 * h2**2)

    values = [base, d_temp(), d_vol(), d2_temp(), d2_vol(), d2_temp_vol()]
    values.extend(
        (_shift_n(state, p, h1) - _shift_n(state, p, -h1)) / (2 * h1)
        for p in range(nc)
    )
    values.extend(
        (
            _shift_n(state, p, h2, T=state["T"] + h2)
            - _shift_n(state, p, h2, T=state["T"] - h2)
            - _shift_n(state, p, -h2, T=state["T"] + h2)
            + _shift_n(state, p, -h2, T=state["T"] - h2)
        )
        / (4 * h2**2)
        for p in range(nc)
    )
    values.extend(
        (
            _shift_n(state, p, h2, V=state["V"] + h2)
            - _shift_n(state, p, h2, V=state["V"] - h2)
            - _shift_n(state, p, -h2, V=state["V"] + h2)
            + _shift_n(state, p, -h2, V=state["V"] - h2)
        )
        / (4 * h2**2)
        for p in range(nc)
    )
    for p in range(nc):
        for q in range(nc):
            samples = []
            for dp in (h2, -h2):
                for dq in (h2, -h2):
                    n = list(state["n"])
                    n[p] += dp
                    n[q] += dq
                    samples.append(_at(state, n=n))
            values.append(
                (samples[0] - samples[1] - samples[2] + samples[3]) / (4 * h2**2)
            )
    return values


def _driver_w6(state, variant):
    """Return a Fortran driver that writes every W6 output of `state`."""
    nc = len(state["n"])

    def fmt(values):
        return ", ".join(f"{value:.17g}d0" for value in values)

    kij = fmt([value for row in state["kij"] for value in row])
    lij = fmt([value for row in state["lij"] for value in row])
    return (
        "program main\n"
        f"  use pr76_{variant}_w6_mod, only: pr76_{variant}_w6\n"
        "  implicit none\n"
        f"  integer, parameter :: nc = {nc}\n"
        "  real(8) :: n(nc), V, T\n"
        "  real(8) :: Pc(nc), Tc(nc), w(nc), kij(nc, nc), lij(nc, nc)\n"
        "  real(8) :: f, dfdT, dfdV, dfdT2, dfdV2, dfdTdV\n"
        "  real(8) :: dfdn(nc), dfdndT(nc), dfdndV(nc), dfdndn(nc, nc)\n"
        "  integer :: i, j\n"
        f"  n = [{fmt(state['n'])}]\n"
        f"  Tc = [{fmt(state['Tc'])}]\n"
        f"  Pc = [{fmt(state['Pc'])}]\n"
        f"  w = [{fmt(state['w'])}]\n"
        f"  kij = reshape([{kij}], [nc, nc])\n"
        f"  lij = reshape([{lij}], [nc, nc])\n"
        f"  V = {state['V']:.17g}d0\n"
        f"  T = {state['T']:.17g}d0\n"
        "  call pr76_" + variant + "_w6(nc, V, T, Pc, Tc, kij, lij, n, w, "
        "f, dfdT, dfdV, dfdn, dfdT2, dfdV2, dfdndn, dfdTdV, dfdndT, "
        "dfdndV)\n"
        "  write(*,*) f, dfdT, dfdV, dfdT2, dfdV2, dfdTdV\n"
        "  write(*,*) (dfdn(i), i=1,nc)\n"
        "  write(*,*) (dfdndT(i), i=1,nc)\n"
        "  write(*,*) (dfdndV(i), i=1,nc)\n"
        "  do i=1,nc\n"
        "    write(*,*) (dfdndn(i,j), j=1,nc)\n"
        "  end do\n"
        "end program main\n"
    )
