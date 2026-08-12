#!/usr/bin/env python3
"""Record provenance for one kernel's batch into a version directory's meta.json.

Called by run_bench.sh once per kernel. Environment facts (git commit, host, GPU,
toolchain versions) are collected here rather than in bash, and merged into any
existing meta.json so several kernels can share one version directory.

Usage:
    record_meta.py <meta.json> --kernel julia --impls rust-gpu,cuda \
        --runs 30 --sizes "7168 9216 11264" --profile release \
        --started <iso8601> --finished <iso8601>
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
    parser.add_argument("--impls", required=True, help="comma-separated impl labels")
    parser.add_argument("--runs", required=True, type=int)
    parser.add_argument("--sizes", required=True, help="space-separated sizes")
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

    meta.setdefault("kernels", {})[args.kernel] = {
        "impls": args.impls.split(","),
        "runs": args.runs,
        "sizes": args.sizes.split(),
        "profile": args.profile,
        "started_at": args.started,
        "finished_at": args.finished,
    }

    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
