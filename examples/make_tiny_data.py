"""Create a deterministic interaction file for the end-to-end smoke experiment."""

import csv
from pathlib import Path


destination = Path(__file__).with_name("tiny_interactions.csv")
with destination.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["user_id", "item_id", "timestamp"])
    writer.writeheader()
    for user in range(1, 25):
        offset = (user * 3) % 31
        for step in range(12):
            item = 1 + ((offset + step * (1 + user % 3)) % 31)
            writer.writerow({"user_id": user, "item_id": item, "timestamp": step})
print(destination)

