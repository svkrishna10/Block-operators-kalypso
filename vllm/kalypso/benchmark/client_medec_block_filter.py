import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import requests

import scenarios


DEFAULT_MEDEC_CSV = (
    Path(__file__).resolve().parent
    / "sample_data"
    / "MEDEC-TrainingSet-1000.csv"
)


def count_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        default=os.environ.get(
            "MODEL",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8003")),
    )

    parser.add_argument(
        "--block-size",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--data",
        default=os.environ.get(
            "MEDEC_CSV",
            str(DEFAULT_MEDEC_CSV),
        ),
    )

    args = parser.parse_args()

    if args.block_size <= 0:
        raise ValueError("block size must be > 0")

    data_path = Path(args.data)
    input_rows = count_rows(data_path)
    expected_blocks = math.ceil(input_rows / args.block_size)

    endpoint = (
        f"http://localhost:{args.port}"
        "/v1/semantic/query"
    )

    payload = {
        "data_path": str(data_path),
        "ops": [
            {
                "op": "block_filter",
                "args": {
                    "prompt": scenarios.MEDEC_ERROR_FILTER,
                    "block_size": args.block_size,
                    "max_tokens": 128,
                },
            }
        ],
        "model_name": args.model_name,
    }

    print("=== Block Filter Benchmark ===")
    print(json.dumps(payload, indent=2))
    print()
    print(f"Input rows: {input_rows}")
    print(f"Block size: {args.block_size}")
    print(f"Expected blocks / LLM calls: {expected_blocks}")
    print()

    started = time.perf_counter()

    response = requests.post(
        endpoint,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=None,
    )

    elapsed = time.perf_counter() - started

    response.raise_for_status()

    result = response.json()

    print("=== Response ===")
    print(
        json.dumps(
            {
                "request_id": result.get("request_id"),
                "num_output_rows": result.get("num_output_rows"),
                "latency_sec": result.get("latency_sec"),
            },
            indent=2,
        )
    )

    print()
    print(f"Client latency: {elapsed:.3f} seconds")
    print(f"Expected LLM calls: {expected_blocks}")
    print(
        "Actual block-call count is printed by BlockFilter on the "
        "server as [block-filter] SUMMARY."
    )


if __name__ == "__main__":
    main()