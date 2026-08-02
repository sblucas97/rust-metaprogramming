#!/usr/bin/env python3
"""Summarize a CSV produced by bench_runs.sh.

Prints mean/stdev/min/max/median grouped by (kernel, impl, size), then --
whenever both a "rust" and a "cuda" row exist for the same (kernel, size) --
a comparison table showing how much slower/faster the Rust DSL is than the
hand-written CUDA for that kernel at that size.

Usage:
    analyze_runs.py <csv_file> [-k KERNEL ...] [-o OUT_FILE]

-k/--kernel can be repeated to only include specific kernels (default: all).
-o/--out saves the same report to a file as well as printing it to stdout.
"""
import argparse
import csv
import statistics
from collections import defaultdict


def build_report(csv_file: str, filter_kernels: set[str]) -> list[str]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    with open(csv_file, newline="") as f:
        for row in csv.DictReader(f):
            if filter_kernels and row["kernel"] not in filter_kernels:
                continue
            key = (row["kernel"], row["impl"], row["size"])
            groups[key].append(float(row["elapsed_ms"]))

    if not groups:
        raise SystemExit("No matching rows.")

    lines: list[str] = []

    cols = ["kernel", "impl", "size", "n", "mean_ms", "stdev_ms", "min_ms", "max_ms", "median_ms"]
    widths = [20, 6, 14, 4, 12, 10, 10, 10, 12]
    lines.append("".join(f"{c:<{w}}" for c, w in zip(cols, widths)))
    lines.append("-" * len(lines[0]))

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

    comparisons = {k: v for k, v in means.items() if "rust" in v and "cuda" in v}
    if comparisons:
        lines.append("")
        lines.append("Rust vs CUDA overhead")
        cols2 = ["kernel", "size", "rust_ms", "cuda_ms", "overhead_ms", "overhead_%"]
        widths2 = [20, 14, 12, 12, 12, 12]
        lines.append("".join(f"{c:<{w}}" for c, w in zip(cols2, widths2)))
        lines.append("-" * len(lines[-1]))
        for (kernel, size), v in sorted(comparisons.items()):
            rust_ms, cuda_ms = v["rust"], v["cuda"]
            overhead_ms = rust_ms - cuda_ms
            overhead_pct = (overhead_ms / cuda_ms * 100) if cuda_ms else float("nan")
            row = [kernel, size, f"{rust_ms:.3f}", f"{cuda_ms:.3f}", f"{overhead_ms:.3f}", f"{overhead_pct:.1f}"]
            lines.append("".join(f"{v:<{w}}" for v, w in zip(row, widths2)))

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_file")
    parser.add_argument("-k", "--kernel", action="append", default=[], help="only include this kernel (repeatable)")
    parser.add_argument("-o", "--out", help="also write the report to this file")
    args = parser.parse_args()

    lines = build_report(args.csv_file, set(args.kernel))
    report = "\n".join(lines)

    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()
