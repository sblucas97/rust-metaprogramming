#!/usr/bin/env python3
"""Summarize CSVs produced by bench_runs.sh.

Prints mean/stdev/min/max/median grouped by (kernel, impl, size), then a
comparison table for every pair of implementations that both have rows for
the same (kernel, size), showing how much slower/faster the first is than
the second. Known impls: "rust-gpu" (the Rust DSL / custom compiler),
"rust" (pure Rust on the CPU, rayon), "cuda" (hand-written CUDA).

Accepts CSV files or directories. A directory is searched recursively for
`runs.csv` (the versioned layout written by run_bench.sh) and `*_runs.csv`
(the older flat layout), so both of these work:

    analyze_runs.py results/latest
    analyze_runs.py results/v3/julia/runs.csv results/v2/julia/runs.csv

Usage:
    analyze_runs.py <path> [<path> ...] [-k KERNEL] [-i IMPL] [-V VERSION] [-o OUT]

-k/-i/-V can each be repeated to restrict the report (default: everything).
-o/--out saves the same report to a file as well as printing it to stdout.
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def expand_paths(paths: list[str]) -> list[Path]:
    """Resolve the given files/directories to a sorted list of CSV files."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            matches = sorted(path.rglob("runs.csv")) + sorted(path.rglob("*_runs.csv"))
            if not matches:
                raise SystemExit(f"No runs.csv found under {path}")
            found.extend(matches)
        elif path.is_file():
            found.append(path)
        else:
            raise SystemExit(f"No such file or directory: {path}")
    # Same file named twice (e.g. a version dir plus one of its CSVs) is harmless
    # to read once, but counting its rows twice would skew every statistic.
    seen: dict[Path, None] = {}
    for path in found:
        seen.setdefault(path.resolve(), None)
    return list(seen)


def build_report(
    paths: list[Path],
    filter_kernels: set[str],
    filter_impls: set[str],
    filter_versions: set[str],
) -> list[str]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    versions: set[str] = set()

    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                # CSVs written before the versioned layout have no version column.
                version = row.get("version") or ""
                if filter_kernels and row["kernel"] not in filter_kernels:
                    continue
                if filter_impls and row["impl"] not in filter_impls:
                    continue
                if filter_versions and version not in filter_versions:
                    continue
                versions.add(version or "(unversioned)")
                key = (row["kernel"], row["impl"], row["size"])
                groups[key].append(float(row["elapsed_ms"]))

    if not groups:
        raise SystemExit("No matching rows.")

    lines: list[str] = []
    lines.append(f"sources: {', '.join(str(p) for p in paths)}")
    lines.append(f"versions: {', '.join(sorted(versions))}")
    lines.append("")

    cols = ["kernel", "impl", "size", "n", "mean_ms", "stdev_ms", "min_ms", "max_ms", "median_ms"]
    widths = [20, 10, 14, 4, 12, 10, 10, 10, 12]
    lines.append("".join(f"{c:<{w}}" for c, w in zip(cols, widths)))
    lines.append("-" * len(lines[-1]))

    means: dict[tuple[str, str], dict[str, float]] = {}
    for (kernel, impl, size), values in sorted(groups.items()):
        n = len(values)
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if n > 1 else 0.0
        row = [
            kernel, impl, size, str(n),
            f"{mean:.3f}", f"{stdev:.3f}",
            f"{min(values):.3f}", f"{max(values):.3f}",
            f"{statistics.median(values):.3f}",
        ]
        lines.append("".join(f"{v:<{w}}" for v, w in zip(row, widths)))
        means.setdefault((kernel, size), {})[impl] = mean

    # One comparison table per pair of impls that co-occur for a (kernel, size).
    # Ordered so each table reads "how much slower is A than the baseline B".
    impl_pairs = [("rust-gpu", "cuda"), ("rust", "rust-gpu"), ("rust", "cuda")]
    for a, b in impl_pairs:
        comparisons = {k: v for k, v in means.items() if a in v and b in v}
        if not comparisons:
            continue
        lines.append("")
        lines.append(f"{a} vs {b} overhead")
        cols2 = ["kernel", "size", f"{a}_ms", f"{b}_ms", "overhead_ms", "overhead_%", "slowdown_x"]
        widths2 = [20, 14, 14, 14, 12, 12, 12]
        lines.append("".join(f"{c:<{w}}" for c, w in zip(cols2, widths2)))
        lines.append("-" * len(lines[-1]))
        for (kernel, size), v in sorted(comparisons.items()):
            a_ms, b_ms = v[a], v[b]
            overhead_ms = a_ms - b_ms
            overhead_pct = (overhead_ms / b_ms * 100) if b_ms else float("nan")
            slowdown = (a_ms / b_ms) if b_ms else float("nan")
            row = [
                kernel, size, f"{a_ms:.3f}", f"{b_ms:.3f}",
                f"{overhead_ms:.3f}", f"{overhead_pct:.1f}", f"{slowdown:.2f}x",
            ]
            lines.append("".join(f"{v:<{w}}" for v, w in zip(row, widths2)))

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="CSV files or directories to read")
    parser.add_argument("-k", "--kernel", action="append", default=[], help="only include this kernel (repeatable)")
    parser.add_argument("-i", "--impl", action="append", default=[], help="only include this impl (repeatable)")
    parser.add_argument("-V", "--version", action="append", default=[], help="only include this version (repeatable)")
    parser.add_argument("-o", "--out", help="also write the report to this file")
    args = parser.parse_args()

    lines = build_report(
        expand_paths(args.paths),
        set(args.kernel),
        set(args.impl),
        set(args.version),
    )
    report = "\n".join(lines)

    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()
