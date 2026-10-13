"""
Sort data_views.json by view key (ascending) and save as
data_views_sorted.json.

Usage:
    python tools/sort_views.py data_views.json [data_views_sorted.json]
"""

import json
import sys

input_file = sys.argv[1] if len(sys.argv) > 1 else "data_views.json"
output_file = sys.argv[2] if len(sys.argv) > 2 else "data_views_sorted.json"

with open(input_file, "r") as f:
    data = json.load(f)

if "views" in data and isinstance(data["views"], list) and data["views"]:
    # Ensure keys are integers (they may be stored as strings)
    for view in data["views"]:
        if isinstance(view.get("key"), str):
            view["key"] = int(view["key"])

    data["views"].sort(key=lambda x: x["key"])

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Sorted, result saved to {output_file}")
else:
    print("Nothing to sort: 'views' is empty or missing")
