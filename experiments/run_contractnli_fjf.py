import json
import time
from pathlib import Path

import requests

from vllm.kalypso.benchmark import scenarios


SERVER = "http://localhost:8003/v1/semantic/query"

BASE = Path(__file__).resolve().parent / "data" / "cnli3"

CONTRACT_DIR = str(BASE / "contracts")
HYPOTHESIS_DIR = str(BASE / "hypotheses")

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

query = {
    "data_path": CONTRACT_DIR,
    "model_name": MODEL,
    "ops": [
        {
            "op": "sem_filter",
            "args": {
                "prompt": scenarios.CONTRACT_NLI_VALID_CONTRACT,
            },
        },
        {
            "op": "join",
            "args": {
                "instruction": scenarios.CONTRACT_NLI_ENTAILMENT_JOIN,
                "right_table": HYPOTHESIS_DIR,
            },
        },
        {
            "op": "sem_filter",
            "args": {
                "prompt": (
                    "Does this contract-hypothesis pair satisfy the "
                    "following semantic condition? "
                    "Answer TRUE only if the relationship is clearly "
                    "supported by the contract.\n\n"
                    "Return only TRUE or FALSE."
                ),
                "max_tokens": 8,
            },
        },
    ],
}

print(json.dumps(query, indent=2))

start = time.perf_counter()

response = requests.post(
    SERVER,
    json=query,
    headers={"Content-Type": "application/json"},
)

elapsed = time.perf_counter() - start

response.raise_for_status()

result = response.json()

print()
print("Request latency:", round(elapsed, 3), "seconds")
print("Output rows:", result.get("num_output_rows"))

print()
print(json.dumps(result, indent=2))
