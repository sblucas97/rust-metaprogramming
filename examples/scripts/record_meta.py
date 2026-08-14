#!/usr/bin/env python3
"""Record provenance for one kernel's batch into a version directory's meta.json.

Called by run_bench.sh once per kernel. Environment facts (git commit, host, GPU,
toolchain versions) are collected here rather than in bash, and merged into any
existing meta.json so several kernels can share one version directory.

Usage:
    record_meta.py <meta.json> --kernel julia \
        --impl-sizes 'rust-gpu=8000 10000' --impl-sizes 'rust=8000' \
        --runs 30 --order blocked --profile release \
        --started <iso8601> --finished <iso8601>

Each kernel entry is a per-impl merge: an --impl-sizes flag names an impl this
batch ran, and only those impls' records are replaced -- a later cuda-oxide-only
batch into the same version updates kernels.<kernel>.by_impl["cuda-oxide"] and
leaves the rust / rust-gpu records (and their timestamps) intact. Entries
written by older versions of this script (flat impls/sizes/runs fields) are
migrated into by_impl on first touch.
"""
import argparse
import json
import os
import platform
import subprocess
from pathlib import Path


def sh(*cmd: str) -> str:
    """Run a command, returning stripped stdout or "" if it isn't available."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def environment(repo_root: Path) -> dict:
    commit = sh("git", "-C", str(repo_root), "rev-parse", "HEAD")
    dirty = bool(sh("git", "-C", str(repo_root), "status", "--porcelain"))
    nvcc = sh("nvcc", "--version").splitlines()
    gpu = sh(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    )
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "hostname": platform.node(),
        "kernel_release": platform.release(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu,
        "nvcc": nvcc[-1] if nvcc else "",
        "rustc": sh("rustc", "--version"),
        "cargo": sh("cargo", "--version"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meta_file")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--runs", required=True, type=int)
    parser.add_argument(
        "--impl-sizes",
        action="append",
        required=True,
        metavar="IMPL=SIZES",
        help="sizes one impl ran, e.g. 'rust=1024 2048' (repeatable)",
    )
    parser.add_argument("--order", default="", help="blocked or interleaved")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--started", required=True)
    parser.add_argument("--finished", required=True)
    args = parser.parse_args()

    meta_path = Path(args.meta_file)
    repo_root = meta_path.resolve().parents[3]

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    meta["version"] = meta_path.parent.name
    # Recorded once per version dir: later kernels in the same batch would only
    # restate the same machine, and a mid-batch commit shouldn't rewrite history.
    meta.setdefault("environment", environment(repo_root))
    meta.setdefault("started_at", args.started)
    meta["finished_at"] = args.finished

    sizes_by_impl = {}
    for spec in args.impl_sizes:
        impl, sep, impl_sizes = spec.partition("=")
        if not impl or not sep:
            raise SystemExit(f"--impl-sizes expects IMPL=SIZES, got {spec!r}")
        sizes_by_impl[impl] = impl_sizes.split()

    entry = meta.setdefault("kernels", {}).setdefault(args.kernel, {})
    by_impl = entry.setdefault("by_impl", {})

    # Migrate entries from the flat format (one record for the whole batch)
    # into per-impl records, so untouched impls keep their provenance.
    if "runs" in entry:
        old_sizes_by_impl = entry.get("sizes_by_impl", {})
        for impl in entry.get("impls", []):
            by_impl.setdefault(impl, {
                "sizes": old_sizes_by_impl.get(impl, entry.get("sizes", [])),
                "runs": entry["runs"],
                "order": entry.get("order", ""),
                "profile": entry.get("profile", ""),
                "started_at": entry.get("started_at", ""),
                "finished_at": entry.get("finished_at", ""),
            })
        for stale in ("runs", "sizes", "sizes_by_impl", "order", "profile",
                      "started_at", "finished_at"):
            entry.pop(stale, None)

    # Replace exactly the impls this batch ran; everything else stays put.
    for impl, sizes in sizes_by_impl.items():
        by_impl[impl] = {
            "sizes": sizes,
            "runs": args.runs,
            "order": args.order,
            "profile": args.profile,
            "started_at": args.started,
            "finished_at": args.finished,
        }
    entry["impls"] = sorted(by_impl)

    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
