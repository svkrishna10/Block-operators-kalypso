#!/usr/bin/env python3

import argparse
import csv
import glob
import math
import re
from collections import defaultdict
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families


# ---------------------------------------------------------------------------
# Prometheus parsing
# ---------------------------------------------------------------------------

def parse_prom_file(path: Path):
    """
    Parse a Prometheus text exposition file.

    Returns:
        dict[
            metric_family_name,
            {
                "samples": [
                    {
                        "name": sample_name,
                        "labels": dict[str, str],
                        "value": float,
                    },
                    ...
                ]
            }
        ]
    """
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8", errors="replace")

    families = {}

    for family in text_string_to_metric_families(text):
        samples = []

        for sample in family.samples:
            samples.append(
                {
                    "name": sample.name,
                    "labels": dict(sample.labels),
                    "value": float(sample.value),
                }
            )

        families[family.name] = {
            "samples": samples,
        }

    return families


def all_samples(metrics):
    for family_name, family in metrics.items():
        for sample in family["samples"]:
            yield family_name, sample


def metric_total(metrics, candidates):
    """
    Return the sum of all samples whose metric name is one of candidates.

    This handles labelled Prometheus counters by summing their samples.
    """
    candidate_set = set(candidates)

    total = 0.0
    found = False

    for _, sample in all_samples(metrics):
        if sample["name"] in candidate_set:
            total += sample["value"]
            found = True

    return total if found else None


def find_metric_by_keywords(metrics, keyword_groups):
    """
    Find the first metric whose sample name contains all keywords from
    one keyword group.

    Example:
        [["prompt", "tokens"], ["generation", "tokens"]]
    """
    for _, sample in all_samples(metrics):
        name = sample["name"].lower()

        for keywords in keyword_groups:
            if all(k in name for k in keywords):
                return sample["name"]

    return None


def metric_sum(metrics, sample_name):
    total = 0.0
    found = False

    for _, sample in all_samples(metrics):
        if sample["name"] == sample_name:
            total += sample["value"]
            found = True

    return total if found else None


# ---------------------------------------------------------------------------
# Counter deltas
# ---------------------------------------------------------------------------

def counter_delta(before, after, candidates):
    b = metric_total(before, candidates)
    a = metric_total(after, candidates)

    if b is None or a is None:
        return None

    delta = a - b

    # Counters should not decrease. If the server restarted between
    # before/after, the delta cannot be interpreted reliably.
    if delta < 0:
        return None

    return delta


# ---------------------------------------------------------------------------
# Request latency
# ---------------------------------------------------------------------------

def find_histogram_components(metrics, base_candidates):
    """
    Find histogram _sum and _count metric names.

    Example:
      vllm:e2e_request_latency_seconds_sum
      vllm:e2e_request_latency_seconds_count
    """
    found = []

    for _, sample in all_samples(metrics):
        name = sample["name"]

        for base in base_candidates:
            if name == base + "_sum":
                found.append((base, "sum", name))
            elif name == base + "_count":
                found.append((base, "count", name))

    return found


def histogram_average_delta(before, after, base_candidates):
    """
    Compute average histogram observation duration during this run:

        delta(sum) / delta(count)
    """
    for base in base_candidates:
        sum_name = base + "_sum"
        count_name = base + "_count"

        b_sum = metric_sum(before, sum_name)
        a_sum = metric_sum(after, sum_name)

        b_count = metric_sum(before, count_name)
        a_count = metric_sum(after, count_name)

        if None in (b_sum, a_sum, b_count, a_count):
            continue

        d_sum = a_sum - b_sum
        d_count = a_count - b_count

        if d_count > 0 and d_sum >= 0:
            return d_sum / d_count

    return None


# ---------------------------------------------------------------------------
# Prefix cache
# ---------------------------------------------------------------------------

def prefix_cache_metrics(before, after):
    query_candidates = [
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_queries",
    ]

    hit_candidates = [
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_hits",
    ]

    queries = counter_delta(before, after, query_candidates)
    hits = counter_delta(before, after, hit_candidates)

    if queries is None:
        # Try discovery by keywords.
        query_name = find_metric_by_keywords(
            after,
            [
                ["prefix", "cache", "queries"],
            ],
        )

        if query_name:
            queries = counter_delta(
                before,
                after,
                [query_name],
            )

    if hits is None:
        hit_name = find_metric_by_keywords(
            after,
            [
                ["prefix", "cache", "hits"],
            ],
        )

        if hit_name:
            hits = counter_delta(
                before,
                after,
                [hit_name],
            )

    hit_rate = None

    if queries is not None and queries > 0 and hits is not None:
        hit_rate = 100.0 * hits / queries

    return queries, hits, hit_rate


# ---------------------------------------------------------------------------
# KV-cache peak
# ---------------------------------------------------------------------------

def peak_kv_usage(timeseries_path: Path):
    """
    Find the largest gauge-like value from metrics whose name appears to
    represent KV-cache usage.

    We support both percentage and 0-1 representations.
    """
    if not timeseries_path.exists():
        return None

    metrics = parse_prom_file(timeseries_path)

    candidates = []

    for _, sample in all_samples(metrics):
        name = sample["name"].lower()

        if "kv" not in name:
            continue

        if "usage" not in name and "util" not in name:
            continue

        value = sample["value"]

        # Ignore NaN/inf.
        if not math.isfinite(value):
            continue

        candidates.append(value)

    if not candidates:
        return None

    peak = max(candidates)

    # Normalize [0,1] utilization to percentage.
    if 0.0 <= peak <= 1.0:
        peak *= 100.0

    return peak


# ---------------------------------------------------------------------------
# Latency from benchmark log
# ---------------------------------------------------------------------------

def parse_latency_from_log(log_path: Path):
    """
    Try common latency strings emitted by the benchmark/wrapper.

    Returns seconds or None.
    """
    if not log_path.exists():
        return None

    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    patterns = [
        r"Request latency:\s*([0-9.]+)\s*seconds",
        r"Request latency:\s*([0-9.]+)\s*s",
        r"Client latency:\s*([0-9.]+)\s*seconds",
        r"Client latency:\s*([0-9.]+)\s*s",
        r"latency_sec[\"']?\s*[:=]\s*([0-9.]+)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)

        if matches:
            return float(matches[-1])

    return None


# ---------------------------------------------------------------------------
# Run processing
# ---------------------------------------------------------------------------

def process_run(log_dir: Path, run_name: str):
    before_path = log_dir / f"{run_name}-metrics-before.prom"
    after_path = log_dir / f"{run_name}-metrics-after.prom"
    timeseries_path = log_dir / f"{run_name}-metrics-timeseries.prom"

    # The wrapper may use either <run>.log or <run>-log.
    possible_logs = [
        log_dir / f"{run_name}.log",
        log_dir / f"{run_name}-log",
    ]

    log_path = next(
        (p for p in possible_logs if p.exists()),
        None,
    )

    before = parse_prom_file(before_path)
    after = parse_prom_file(after_path)

    # Requests.
    requests = counter_delta(
        before,
        after,
        [
            "vllm:request_success_total",
            "vllm:num_requests_success_total",
            "vllm:request_success",
        ],
    )

    # Prompt tokens.
    prompt_tokens = counter_delta(
        before,
        after,
        [
            "vllm:prompt_tokens_total",
            "vllm:prompt_tokens",
        ],
    )

    # Generated tokens.
    generated_tokens = counter_delta(
        before,
        after,
        [
            "vllm:generation_tokens_total",
            "vllm:generation_tokens",
        ],
    )

    # Prefix cache.
    prefix_queries, prefix_hits, prefix_hit_rate = prefix_cache_metrics(
        before,
        after,
    )

    # E2E request latency from a Prometheus histogram, if exposed.
    average_latency_prom = histogram_average_delta(
        before,
        after,
        [
            "vllm:e2e_request_latency_seconds",
            "vllm:request_latency_seconds",
            "vllm:request_e2e_latency_seconds",
        ],
    )

    # Latency printed by the benchmark/wrapper is preferred if available.
    latency_log = parse_latency_from_log(log_path) if log_path else None

    latency = latency_log
    latency_source = "benchmark log"

    if latency is None:
        latency = average_latency_prom
        latency_source = "Prometheus histogram"

    peak_kv = peak_kv_usage(timeseries_path)

    return {
        "run": run_name,
        "latency_sec": latency,
        "latency_source": latency_source if latency is not None else "",
        "llm_requests": requests,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "prefix_cache_queries": prefix_queries,
        "prefix_cache_hits": prefix_hits,
        "prefix_cache_hit_rate_pct": prefix_hit_rate,
        "peak_kv_cache_usage_pct": peak_kv,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(value, digits=3):
    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:.{digits}f}"

    return str(value)


def markdown_table(rows):
    headers = [
        "Run",
        "Latency (s)",
        "LLM requests",
        "Prompt tokens",
        "Generated tokens",
        "Prefix queries",
        "Prefix hits",
        "Prefix hit rate (%)",
        "Peak KV usage (%)",
    ]

    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append(
        "| "
        + " | ".join(["---"] * len(headers))
        + " |"
    )

    for row in rows:
        values = [
            row["run"],
            fmt(row["latency_sec"]),
            fmt(row["llm_requests"], 0),
            fmt(row["prompt_tokens"], 0),
            fmt(row["generated_tokens"], 0),
            fmt(row["prefix_cache_queries"], 0),
            fmt(row["prefix_cache_hits"], 0),
            fmt(row["prefix_cache_hit_rate_pct"]),
            fmt(row["peak_kv_cache_usage_pct"]),
        ]

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def csv_value(value):
    if value is None:
        return ""

    return value


def write_csv(path: Path, rows):
    fields = [
        "run",
        "latency_sec",
        "latency_source",
        "llm_requests",
        "prompt_tokens",
        "generated_tokens",
        "prefix_cache_queries",
        "prefix_cache_hits",
        "prefix_cache_hit_rate_pct",
        "peak_kv_cache_usage_pct",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: csv_value(row.get(key))
                    for key in fields
                }
            )


def write_markdown(path: Path, title: str, rows):
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(markdown_table(rows))
        f.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build baseline metrics from Prometheus run files."
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("experiments/logs"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results"),
    )

    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help=(
            "Run names, e.g. "
            "medec16-run1 medec16-run2 medec16-run3"
        ),
    )

    parser.add_argument(
        "--title",
        default="Baseline Metrics",
    )

    parser.add_argument(
        "--csv",
        default="baseline_metrics.csv",
    )

    parser.add_argument(
        "--md",
        default="baseline_metrics.md",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for run_name in args.runs:
        print(f"Processing {run_name} ...")

        try:
            row = process_run(
                args.log_dir,
                run_name,
            )
        except FileNotFoundError as exc:
            print(f"ERROR: missing file: {exc}")
            raise SystemExit(1)

        rows.append(row)

        print(
            "  "
            f"requests={row['llm_requests']} "
            f"prompt={row['prompt_tokens']} "
            f"generated={row['generated_tokens']} "
            f"prefix_hit_rate="
            f"{row['prefix_cache_hit_rate_pct']}"
        )

    csv_path = args.output_dir / args.csv
    md_path = args.output_dir / args.md

    write_csv(csv_path, rows)
    write_markdown(md_path, args.title, rows)

    print()
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print()
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
