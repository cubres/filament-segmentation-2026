"""Render a new chart from the recorded local canary; no invented measurements."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
receipt = json.loads((root / "evidence/local-canary.json").read_text())
with (root / "evidence/local-validation.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
expected = {item["config"]: item["pq"] for item in receipt["validation"]}
assert receipt["official_score"] is None
assert len(rows) == len(expected)
assert all(float(row["pq"]) == expected[row["config"]] for row in rows)
assert all(int(row["images"]) == 4 for row in rows)
figure, axes = plt.subplots(figsize=(8, 3.8), facecolor="#FFF9EE")
axes.set_facecolor("#FFF9EE")
labels = {"fixed_threshold": "Fixed threshold", "core_growth_050_130": "Dark-core growth"}
values = [float(row["pq"]) for row in rows]
bars = axes.barh([labels[row["config"]] for row in rows], values, color=["#A3ABB4", "#D1765A"])
axes.bar_label(bars, labels=[f"{value:.6f}" for value in values], padding=5)
axes.set_xlim(0, max(values) * 1.35)
axes.set_xlabel("Local panoptic quality (PQ)")
axes.set_title("Four held-out images; no official score", loc="left")
axes.spines[["top", "right"]].set_visible(False)
figure.tight_layout()
with (root / "assets/local-validation.png").open("xb") as stream:
    figure.savefig(stream, format="png", dpi=160)
plt.close(figure)
