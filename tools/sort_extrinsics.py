"""
Sort data_extrinsics.json by pose key (ascending) and save as
data_extrinsics_sorted.json.

Usage:
    python tools/sort_extrinsics.py data_extrinsics.json [data_extrinsics_sorted.json]
"""

import json
import sys

input_file = sys.argv[1] if len(sys.argv) > 1 else "data_extrinsics.json"
output_file = sys.argv[2] if len(sys.argv) > 2 else "data_extrinsics_sorted.json"

with open(input_file, "r") as f:
    data = json.load(f)

if "extrinsics" in data and isinstance(data["extrinsics"], list) and data["extrinsics"]:
    # Ensure keys are integers (they may be stored as strings, e.g. "47")
    for ext in data["extrinsics"]:
        if isinstance(ext.get("key"), str):
            ext["key"] = int(ext["key"])

    data["extrinsics"].sort(key=lambda x: x["key"])

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Sorted, result saved to {output_file}")
else:
    print("Nothing to sort: 'extrinsics' is empty or missing")
