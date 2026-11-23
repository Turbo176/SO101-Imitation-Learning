# Project history and attribution

This project is based on LeRobot 0.3.4 at commit `882c80d446a63a44868c67ae535467af32ce0e80`. The public repository contains a source snapshot with the SO-101 changes. Upstream copyright headers and Apache-2.0 notices remain in place; the original source and history are available in [LeRobot](https://github.com/huggingface/lerobot/tree/882c80d446a63a44868c67ae535467af32ce0e80).

## Experiment dates

| Date | Record |
| --- | --- |
| July–September 2026 | Modification times of local camera, serial and SO-101 integration files |
| September 29, 2026 | ACT loss CSV, checkpoint state and deployment videos |
| October 7, 2026 | SmolVLA training CSV with UTC timestamps, through 50,000 steps |
| October 7–13, 2026 | Offline inference tools, saved predictions and analysis exports |
| November 19, 2026 | Public source release, setup/training/deployment guides and video layouts |

ACT terminal timestamps have no explicit timezone. Video/file modification times support the timeline but do not independently authenticate capture dates. Source names, hashes and preserved times are in [source_artifacts.json](../../results/so101/source_artifacts.json) and [source_snapshot_dates.json](../../results/so101/source_snapshot_dates.json).

## Changes in this repository

| Area | Implementation |
| --- | --- |
| OpenCV capture | Optional FOURCC and requested-mode handling |
| SO-101 serial | Position-read and startup-configuration retries, propagated through leader/follower configs |
| Recording | Skip an empty episode on early exit |
| Policy configuration | Windows temporary-file compatibility and saved-config fields |
| Robot registration | Existing dual-SO101 integration; not evaluated in the single-arm experiment |
| Deployment tools | Local FP32 SmolVLA loading, processor assets, camera-key mapping, queued actions and disconnect cleanup |
| Results | Original loss exports, saved prediction arrays, metric recomputation and video/joint visualization |

The deployment scripts were moved from a local output directory into `scripts/so101/`. Their defaults now use relative paths and live mode requires an explicit port. The runner also fixes duplicate chunk computation and cleanup after startup failures. Offline checks cover these changes; the revised runner has not had a new physical evaluation.

Checkpoint/config identities are in [checkpoint_manifest.json](../../results/so101/checkpoint_manifest.json). Archived configs replace machine paths and select PyAV; fresh recipes also reset resume/evaluation fields and explicitly load the SmolVLA backbone. Original hashes identify the source files separately.

## Third-party material

| Material | Source and license handling |
| --- | --- |
| LeRobot | Hugging Face and contributors, Apache-2.0; original headers retained |
| ACT | LeRobot implementation of [ALOHA / ACT](https://tonyzhaozh.github.io/aloha/) |
| SmolVLA | LeRobot implementation; [paper](https://huggingface.co/papers/2506.01844) |
| SmolVLM2 assets | Upstream model declares Apache-2.0; downloaded separately at the documented revision |
| ResNet-18 weights | Torchvision download, excluded from Git |
| Data and videos | This physical experiment; selected recordings and result samples included |
| Diagrams and plots | Local SVG assets and plots from the experiment's data |

The source license is [Apache-2.0](../../LICENSE). Full datasets, trained weights, environments, physical calibration and private server settings are excluded. Downloaded assets retain their upstream terms. The Python package keeps the `lerobot` name for import compatibility; this repository does not publish it to PyPI.
