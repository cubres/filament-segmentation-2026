# Validation record

## Small local canary

Date: 2026-10-23.

- Four calibration images select one of five fixed settings.
- Four held-out images come from different calendar months.
- One test image checks the output contract.
- The canary reads segmentation masks only. It does not use spines, boxes, or class labels.

The selected setting is `core_growth_050_130`.
Its calibration PQ is 0.099541. Its held-out PQ is 0.108615.
The control reaches 0.052726 on calibration and 0.080914 on validation.
These are local values from the public host-reference metric.
They are not official Kaggle scores.

[Calibration values](../evidence/local-calibration.csv), [validation values](../evidence/local-validation.csv), and the [split](../evidence/canary-split.json) are retained.
The [canary receipt](../evidence/local-canary.json) binds the test to its source hash.
The source and datasets can change, so always compare the recorded hashes.

## Notebook execution

All seven code cells executed with bounded input adapters.
The adapters substitute eight local training images and one test image for the full dataset.
They also substitute the preselected four-plus-four split and a separate output folder.
The canonical notebook stays unchanged during the check.

The run rendered six plots. It took about 8.56 seconds on this local setup.
This time is not a full-dataset runtime forecast.
Competition-image panels are not redistributed in this repository.
The [notebook execution receipt](../evidence/notebook-canary.json) records the scope.

## Contract checks

Seven public tests use synthetic arrays and no network.
They check COCO RLE round trips, sparse-mask encoding, perfect and empty PQ cases, dark-core rejection, month separation, overlapping masks, and foreign image IDs.

A separate local comparison checked the public host's PQ semantics on a synthetic IoU matrix.
That comparison needs the host reference notebook and PyTorch.
The host source is not redistributed here, so it is not part of the seven public tests.

## What this does not prove

Four held-out images are too few for a strong performance claim.
The local check does not establish generalization, competition rank, or a complete valid submission.
The full Kaggle run must complete before a notebook-linked submission is possible.
An official result needs the exact completed submission, notebook version, file, and visible score badge.
