# Validation record

## Completed Kaggle notebook Version 2

The [public notebook](https://www.kaggle.com/code/prvsiyan/notebook24a4dce69e) completed Version 2 on 2026-10-23.
The exact status read at 17:36 UTC was `complete`.
This is a notebook-run status, not a competition-submission status.

The run found 707 available training images. It selected the setting on 24 images and held out 16 images from separate months.
Some images have several annotators. The held-out metric contains 21 annotator records.
The selected setting is `core_growth_050_130`.

| Method | Held-out PQ |
|---|---:|
| Fixed threshold | 0.06356608798040476 |
| Dark-core growth | 0.08902624179349382 |

All 180 test images ran. The exported file contains 5,316 predicted instance masks.
The recorded runtime is 375.652268073 seconds.
These facts establish a completed artifact run. They do not establish an official score or leaderboard rank.

The accepted server source has the same cell types and cell contents as the reviewed candidate.
The whole-file hashes differ because server notebook metadata differs.
Both hashes are recorded in [the version-specific evidence](../evidence/kaggle-v2-run.json).
The record also keeps package versions and the submission-file hash.
It omits private paths, logs, competition images, labels, and predicted masks.

The output validator decoded all 5,316 masks from all 180 test images.
The [sanitized receipt](../evidence/kaggle-v2-output-validation.json) keeps the exact version and output hashes.
Its scope is artifact validation against the supplied capture receipt.
That check does not independently authenticate the receipt or establish a competition score.

Submission 57568420 was accepted from the notebook's Version 2 menu.
At 17:49 UTC on 2026-10-23, its [recorded status](../evidence/kaggle-v2-submission-status-1749.json) remained `PENDING`.
The official score field was empty. No native notebook score badge has been verified.
The accepted attempt must not be repeated while its outcome remains pending.

The earlier four-image canary below remains a separate historical check.
Its sample differs, so do not interpret the two PQ tables as an improvement over time.

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
The full Kaggle run is complete. Its notebook-linked submission was accepted and remains pending in the latest recorded check.
An official result needs the exact completed submission, notebook version, file, and visible score badge.
