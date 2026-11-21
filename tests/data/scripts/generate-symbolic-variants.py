#!/usr/bin/env python3
"""Generate the thermodiff symbolic variant files for the benchmark.

One ``.f90`` file per (workload x variant) plus ``manifest.json`` per
variant directory, under ``bench/src/generated/<model>_<variant>_<token>/``
(consumer-side interface contract, frozen file set).  The generation
metrics required by ``reports/protocol.md`` section 8.2 are captured
through :mod:`benchmarks.td_bench`: generation wall time and sub-phase
times, path/lines/bytes of each file, and the symbolic operation count.

Usage (from the session root):

    .venv/bin/python scripts/generate-symbolic-variants.py

The generated files are deterministic (no timestamps inside them); the
manifests additionally record the measured generation times, so they are
not byte-stable between runs by design.
"""

import argparse
import json
import os
import sys

SESSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SESSION_ROOT, "thermodiff"))
sys.path.insert(0, os.path.dirname(os.path.dirname(SESSION_ROOT)))

from benchmarks.td_bench import (  # noqa: E402
    WORKLOADS,
    generate_workload,
    variant_settings,
)

#: Order in which the workloads are generated (frozen file names).
WORKLOAD_ORDER = [
    "w0",
    "w1",
    "w2",
    "w3",
    "w4",
    "w5a",
    "w5b",
    "w5c",
    "w5d",
    "w6",
]

#: Manifest keys copied verbatim from the hook metadata (protocol 8.2 maps
#: ``lines -> code_lines``, ``bytes -> code_bytes``,
#: ``operation_count -> op_count``, ``operation_count_tool -> op_count_tool``).
_ENTRY_KEYS = [
    "workload",
    "derivatives",
    "subroutine_name",
    "variant",
    "variant_id",
    "variant_token",
    "optional_derivatives",
    "array_cse",
    "share",
    "spliced_uses",
    "generation_time_s",
    "parse_time_s",
    "diff_time_s",
    "emit_time_s",
    "lines",
    "bytes",
    "operation_count",
    "operation_count_tool",
    "n_temporaries",
]


def _manifest_entry(metadata, out_dir):
    """Return the manifest copy of one hook metadata dict."""
    entry = {key: metadata[key] for key in _ENTRY_KEYS}
    entry["file"] = os.path.basename(metadata["path"])
    entry["path"] = os.path.relpath(metadata["path"], out_dir)
    return entry


def generate_all(model, out_root, variants, workloads, optional_derivatives):
    """Generate every requested (workload x variant) file.

    Returns
    -------
    list of dict
        The hook metadata of every generated file, in generation order.
    """
    model_name = os.path.splitext(os.path.basename(model))[0]
    results = []
    for variant in variants:
        settings = variant_settings(variant)
        out_dir = os.path.join(
            out_root,
            f"{model_name}_{settings['key']}_{settings['token']}",
        )
        os.makedirs(out_dir, exist_ok=True)
        entries = []
        for workload in workloads:
            metadata = generate_workload(
                model,
                workload,
                variant,
                out_dir,
                optional_derivatives=optional_derivatives,
            )
            results.append(metadata)
            entries.append(_manifest_entry(metadata, out_root))
            print(
                f"{metadata['path']}  "
                f"gen={metadata['generation_time_s']:.3f}s "
                f"(parse={metadata['parse_time_s']:.3f} "
                f"diff={metadata['diff_time_s']:.3f} "
                f"emit={metadata['emit_time_s']:.3f}) "
                f"lines={metadata['lines']} bytes={metadata['bytes']} "
                f"ops={metadata['operation_count']} "
                f"temps={metadata['n_temporaries']}"
            )
        manifest = {
            "model": model_name,
            "model_file": os.path.relpath(model, out_root),
            "variant": {
                "key": settings["key"],
                "id": settings["id"],
                "token": settings["token"],
                "label": settings["label"],
                "array_cse": settings["array_cse"],
                "share": settings["share"],
            },
            "optional_derivatives": bool(optional_derivatives),
            "operation_count_tool": "sympy.count_ops(visual=False)",
            "files": entries,
        }
        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"wrote {manifest_path}")

    aggregate = {
        "model": model_name,
        "model_file": os.path.relpath(model, out_root),
        "variants": [
            {
                "key": variant_settings(v)["key"],
                "id": variant_settings(v)["id"],
                "token": variant_settings(v)["token"],
                "array_cse": variant_settings(v)["array_cse"],
                "share": variant_settings(v)["share"],
            }
            for v in variants
        ],
        "optional_derivatives": bool(optional_derivatives),
        "operation_count_tool": "sympy.count_ops(visual=False)",
        "files": [_manifest_entry(metadata, out_root) for metadata in results],
    }
    aggregate_path = os.path.join(out_root, "manifest.json")
    with open(aggregate_path, "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {aggregate_path}")
    return results


def main(argv=None):
    """Run the driver.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate the thermodiff symbolic variant files "
            "(workload x variant) for the benchmark."
        )
    )
    parser.add_argument(
        "--model",
        default=os.path.join(SESSION_ROOT, "models", "pr76", "pr76.model"),
        help="path to the .model file (default: models/pr76/pr76.model)",
    )
    parser.add_argument(
        "--out-root",
        default=os.path.join(SESSION_ROOT, "bench", "src", "generated"),
        help="output root (default: bench/src/generated)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["v1", "v2", "v3"],
        help="variants to generate (default: v1 v2 v3)",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=list(WORKLOAD_ORDER),
        help="workload ids to generate (default: all ten)",
    )
    parser.add_argument(
        "--no-optional-derivatives",
        dest="optional_derivatives",
        action="store_false",
        help=(
            "declare outputs as plain intent(out) instead of optional "
            "(the benchmark uses optional_derivatives=True per D-014)"
        ),
    )
    args = parser.parse_args(argv)

    unknown = [w for w in args.workloads if w.lower() not in WORKLOADS]
    if unknown:
        print(f"error: unknown workloads {unknown}", file=sys.stderr)
        return 1

    results = generate_all(
        args.model,
        args.out_root,
        args.variants,
        [w.lower() for w in args.workloads],
        args.optional_derivatives,
    )
    total = sum(r["generation_time_s"] for r in results)
    print(f"generated {len(results)} files in {total:.3f}s (sum of gen times)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
