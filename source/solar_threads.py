"""Original CPU solar-filament baseline. Competition masks are the only labels.

MIT license; copyright (c) 2026 prvsiyan / cubres.
The evaluator follows the public host reference. It is not Kaggle's private scorer.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import time

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask


@dataclass(frozen=True)
class Config:
    name: str = "fixed_threshold"
    low: float = 0.075
    high: float = 0.075
    blur_sigma: float = 18.0
    denoise_sigma: float = 1.6
    min_area: int = 200
    min_core_area: int = 30
    border: int = 24
    closing_size: int = 3


BASELINE = Config()
CANDIDATES = [
    BASELINE,
    Config("core_growth_050_100", low=0.050, high=0.100),
    Config("core_growth_060_110", low=0.060, high=0.110),
    Config("core_growth_070_120", low=0.070, high=0.120),
    Config("core_growth_050_130", low=0.050, high=0.130),
]


def encode_mask(mask: np.ndarray) -> dict:
    return coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (2048, 2048):
        raise ValueError(f"Expected one 2048 by 2048 image: {path.name}")
    return image


def image_features(image: np.ndarray, config: Config = BASELINE) -> tuple[np.ndarray, np.ndarray]:
    """Measure dark local contrast. Erode the largest bright disk at its edge."""
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Input must be a finite grayscale image")
    bright = (image > 30).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    disk = np.zeros(image.shape, np.uint8)
    if count > 1:
        disk[labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))] = 1
    # Fill dark holes in the solar disk before the edge is eroded.
    contours, _ = cv2.findContours(disk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(disk, contours, -1, 1, thickness=cv2.FILLED)
    radius = max(1, config.border)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    disk = cv2.erode(disk, kernel).astype(bool)
    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), config.blur_sigma)
    denoised = cv2.GaussianBlur(image.astype(np.float32), (0, 0), config.denoise_sigma)
    contrast = (smooth - denoised) / np.maximum(smooth, 24.0)
    contrast[~disk] = 0
    return contrast, disk


def predict_features(contrast: np.ndarray, disk: np.ndarray, config: Config) -> np.ndarray:
    """Keep a weak dark component only when it contains a strong dark core.

    This seed test is our ablation. Output instances are disjoint by construction.
    The method can merge nearby filaments. That error is visible in PQ.
    """
    weak = ((contrast >= config.low) & disk).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.closing_size, config.closing_size))
    weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, kernel)
    weak[~disk] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    strong = (contrast >= config.high) & disk
    core_counts = np.bincount(labels[strong], minlength=count)
    keep = (stats[:, cv2.CC_STAT_AREA] >= config.min_area) & (core_counts >= config.min_core_area)
    keep[0] = False
    mapping = np.zeros(count, np.int32)
    mapping[np.flatnonzero(keep)] = np.arange(1, int(keep.sum()) + 1)
    return mapping[labels]


def predict_image(image: np.ndarray, config: Config) -> np.ndarray:
    return predict_features(*image_features(image, config), config)


def labels_to_rles(labels: np.ndarray) -> list[dict]:
    """Encode sparse label positions once; avoid one full-image scan per instance."""
    height, width = labels.shape
    flat = labels.ravel(order="F")
    positions = np.flatnonzero(flat)
    if not len(positions):
        return []
    order = np.argsort(flat[positions], kind="stable")
    positions = positions[order]
    values = flat[positions]
    splits = np.flatnonzero(np.diff(values)) + 1
    rles = []
    for pixels in np.split(positions, splits):
        starts = np.r_[0, np.flatnonzero(np.diff(pixels) != 1) + 1]
        ends = np.r_[starts[1:] - 1, len(pixels) - 1]
        run_starts, run_ends = pixels[starts], pixels[ends] + 1
        gaps = run_starts - np.r_[0, run_ends[:-1]]
        lengths = run_ends - run_starts
        counts = np.column_stack([gaps, lengths]).ravel().tolist()
        if run_ends[-1] < flat.size:
            counts.append(int(flat.size - run_ends[-1]))
        rles.append(coco_mask.frPyObjects({"size": [height, width], "counts": counts}, height, width))
    return rles


def annotation_rle(annotation: dict, height: int = 2048, width: int = 2048) -> dict:
    # Deliberately use segmentation only: no spines, boxes, or class labels.
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))
    if isinstance(segmentation["counts"], list):
        return coco_mask.frPyObjects(segmentation, height, width)
    rle = dict(segmentation)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    return rle


def pq_counts(ground_truth: list[dict], predictions: list[dict]) -> dict:
    """Host-reference semantics: all pairs above IoU 0.5; no Hungarian rewrite."""
    ng, npred = len(ground_truth), len(predictions)
    if ng == 0 or npred == 0:
        return {"iou_sum": 0.0, "tp": 0, "fp": npred, "fn": ng}
    iou = coco_mask.iou(predictions, ground_truth, [0] * ng).T
    hits = iou > 0.5
    return {
        "iou_sum": float(iou[hits].sum()),
        "tp": int(hits.sum()),
        "fp": int((~hits.any(axis=0)).sum()),
        "fn": int((~hits.any(axis=1)).sum()),
    }


def aggregate_pq(rows: list[dict]) -> float:
    denominator = sum(row["tp"] + 0.5 * row["fp"] + 0.5 * row["fn"] for row in rows)
    return sum(row["iou_sum"] for row in rows) / denominator if denominator else 0.0


def stable_key(text: str) -> str:
    return hashlib.sha256(("solar-threads-v1:" + text).encode()).hexdigest()


def split_names(names: list[str], calibration_limit: int = 24, validation_limit: int = 16) -> tuple[list[str], list[str]]:
    """A fixed calendar-month split keeps same-date and annotator copies together."""
    unique = sorted(set(names), key=stable_key)
    calibration = [name for name in unique if int(stable_key(name[:6])[:8], 16) % 4 != 0]
    validation = [name for name in unique if int(stable_key(name[:6])[:8], 16) % 4 == 0]
    if not calibration or not validation:
        raise ValueError("Both month groups need images. Do not improvise a split after seeing scores.")
    calibration, validation = calibration[:calibration_limit], validation[:validation_limit]
    assert not {n[:6] for n in calibration} & {n[:6] for n in validation}
    return calibration, validation


def load_annotations(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text())
    by_name = defaultdict(list)
    by_id = defaultdict(list)
    for item in data["images"]:
        by_name[Path(item["file_name"]).name].append(item["id"])
    for annotation in data["annotations"]:
        # Restrict retained fields to the allowed mask task.
        by_id[annotation["image_id"]].append({"segmentation": annotation["segmentation"]})
    return dict(by_name), dict(by_id)


def evaluate(paths: dict[str, Path], names: list[str], by_name: dict, by_id: dict,
             configs: list[Config]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for index, name in enumerate(names):
        image = read_image(paths[name])
        features = image_features(image)
        groups = {image_id: [annotation_rle(a) for a in by_id[image_id]] for image_id in by_name[name]}
        for config in configs:
            start = time.perf_counter()
            labels = predict_features(*features, config)
            predictions = labels_to_rles(labels)
            seconds = time.perf_counter() - start
            for image_id, truth in groups.items():
                rows.append({"config": config.name, "image": name, "annotator_image": image_id,
                             "prediction_count": len(predictions), "seconds": seconds,
                             **pq_counts(truth, predictions)})
        if (index + 1) % 4 == 0 or index + 1 == len(names):
            print(f"Measured {index + 1}/{len(names)} images", flush=True)
    detail = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {"config": config.name, "pq": aggregate_pq(detail[detail.config == config.name].to_dict("records")),
         "images": len(names), "annotator_records": int((detail.config == config.name).sum())}
        for config in configs
    ])
    return summary, detail


def discover(root: Path) -> tuple[Path, dict[str, Path], list[Path]]:
    annotations = list(root.rglob("*Annotations*train*.json"))
    if len(annotations) != 1:
        raise ValueError(f"Expected one official training annotation file, found {len(annotations)}")
    train = {p.name: p for p in root.rglob("*.jpeg") if "train_images" in p.parts}
    test = sorted(p for p in root.rglob("*.jpeg") if "test_images" in p.parts)
    if not train or not test:
        raise ValueError("Attach the official competition data. Train and test images are required.")
    if len(test) != 180:
        raise ValueError(f"Expected the current 180-image test release, found {len(test)}; review data drift.")
    return annotations[0], train, test


def validate_submission(frame: pd.DataFrame, test_stems: set[str], decode_all: bool = True) -> dict:
    if list(frame.columns) != ["filament_id", "segmentation_rle"]:
        raise ValueError("Submission columns changed")
    if frame.empty or frame.filament_id.duplicated().any() or frame.isna().any().any():
        raise ValueError("Empty, duplicate, or missing submission values")
    seen = set()
    rows = 0
    for stem, group in frame.groupby(frame.filament_id.str.rsplit("_", n=1).str[0]):
        if stem not in test_stems:
            raise ValueError(f"Unknown test image: {stem}")
        seen.add(stem)
        coverage = np.zeros((2048, 2048), dtype=bool)
        for counts in group.segmentation_rle:
            if not isinstance(counts, str) or not counts or "\n" in counts:
                raise ValueError("Invalid COCO RLE string")
            rle = {"size": [2048, 2048], "counts": counts.encode("ascii")}
            if decode_all:
                mask = coco_mask.decode(rle).astype(bool)
                if not mask.any() or (coverage & mask).any():
                    raise ValueError("Empty or overlapping predicted masks")
                coverage |= mask
            rows += 1
    # A valid image may have no predicted filaments. Record it, never invent a mask.
    return {"rows": rows, "represented_images": len(seen),
            "zero_prediction_images": sorted(test_stems - seen), "decoded_all": decode_all}


def infer(test_paths: list[Path], config: Config, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, audit = [], []
    for index, path in enumerate(test_paths):
        started = time.perf_counter()
        labels = predict_image(read_image(path), config)
        rles = labels_to_rles(labels)
        for instance, rle in enumerate(rles, 1):
            rows.append({"filament_id": f"{path.stem}_{instance}",
                         "segmentation_rle": rle["counts"].decode("ascii")})
        audit.append({"image_id": path.stem, "filaments": len(rles),
                      "seconds": time.perf_counter() - started,
                      "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if (index + 1) % 20 == 0 or index + 1 == len(test_paths):
            print(f"Predicted {index + 1}/{len(test_paths)} test images", flush=True)
    frame = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    checks = validate_submission(frame, {p.stem for p in test_paths})
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "submission.csv", index=False)
    pd.DataFrame(audit).to_csv(output_dir / "inference-audit.csv", index=False)
    (output_dir / "submission-checks.json").write_text(json.dumps(checks, indent=2))
    return frame, pd.DataFrame(audit)


def run(root: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path, train_paths, test_paths = discover(root)
    by_name, by_id = load_annotations(annotation_path)
    available = sorted(set(train_paths) & set(by_name))
    calibration, validation = split_names(available)
    print(f"{len(available)} train images. {len(test_paths)} test images.")
    print(f"Fit thresholds on {len(calibration)} images. Hold out {len(validation)} images.")
    cal_summary, cal_detail = evaluate(train_paths, calibration, by_name, by_id, CANDIDATES)
    best_name = cal_summary.sort_values(["pq", "config"], ascending=[False, True]).iloc[0].config
    selected = next(config for config in CANDIDATES if config.name == best_name)
    validation_configs = [BASELINE] if selected == BASELINE else [BASELINE, selected]
    val_summary, val_detail = evaluate(train_paths, validation, by_name, by_id, validation_configs)
    for filename, table in [("calibration-summary.csv", cal_summary), ("calibration-detail.csv", cal_detail),
                            ("validation-summary.csv", val_summary), ("validation-detail.csv", val_detail)]:
        table.to_csv(output_dir / filename, index=False)
    manifest = {"calibration": calibration, "validation": validation,
                "split_unit": "calendar_month", "selected_without_validation": asdict(selected)}
    (output_dir / "split-manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Selected from calibration only:", selected.name)
    print(val_summary.to_string(index=False))
    submission, timing = infer(test_paths, selected, output_dir)
    result = {"kind": "local_holdout_and_submission_artifact", "official_score": None,
              "selected_config": asdict(selected), "local_validation": val_summary.to_dict("records"),
              "train_images_available": len(available), "test_images": len(test_paths),
              "submission_rows": len(submission), "runtime_seconds": time.perf_counter() - started,
              "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
              "submission_sha256": hashlib.sha256((output_dir / "submission.csv").read_bytes()).hexdigest(),
              "packages": {name: importlib.metadata.version(name) for name in
                           ["numpy", "pandas", "pycocotools", "matplotlib"]}}
    (output_dir / "run-report.json").write_text(json.dumps(result, indent=2))
    return result
