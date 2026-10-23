"""Render the completed Version 2 held-out measurements, not an official score."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
receipt = json.loads((root / "evidence/kaggle-v2-run.json").read_text())
with (root / "evidence/kaggle-v2-holdout.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
assert receipt["version_number"] == 2 and receipt["run_status"] == "complete"
expected = {item["config"]: item for item in receipt["run_report"]["local_validation"]}
assert len(rows) == len(expected)
assert all(float(row["pq"]) == expected[row["config"]]["pq"] for row in rows)
assert all(int(row["images"]) == 16 and int(row["annotator_records"]) == 21 for row in rows)
figure, axes = plt.subplots(figsize=(8, 3.8), facecolor="#FFF9EE")
axes.set_facecolor("#FFF9EE")
labels = {"fixed_threshold": "Fixed threshold", "core_growth_050_130": "Dark-core growth"}
values = [float(row["pq"]) for row in rows]
bars = axes.barh([labels[row["config"]] for row in rows], values, color=["#A3ABB4", "#D1765A"])
axes.bar_label(bars, labels=[f"{value:.6f}" for value in values], padding=5)
axes.set_xlim(0, max(values) * 1.4)
axes.set_xlabel("Held-out panoptic quality (PQ)")
axes.set_title("Kaggle Version 2: 16 held-out images", loc="left")
figure.text(0.02, 0.01, "Training-data holdout; not an official competition score", fontsize=9, color="#555555")
axes.spines[["top", "right"]].set_visible(False)
figure.tight_layout(rect=(0, 0.05, 1, 1))
with (root / "assets/kaggle-v2-holdout.png").open("xb") as stream:
    figure.savefig(stream, format="png", dpi=160)
plt.close(figure)
