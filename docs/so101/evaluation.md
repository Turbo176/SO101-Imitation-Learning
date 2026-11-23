# Results and recordings

## Recompute the diagnostics

```bash
python scripts/so101/verify_results.py
python -m pip install -e ".[tools]" -c requirements.txt
python scripts/so101/plot_results.py
```

`checkpoint_predictions.npz` stores ACT/SmolVLA predictions for five checkpoints, 15 observations, 50 future steps and six joints, plus targets, a hold-state baseline and normalization scales. The verifier reproduces the saved metrics with zero numerical difference.

The ACT loss CSV contains rounded terminal exports every 200 steps. SmolVLA's original CSV contains 100-step windows; the 200-step display CSV averages adjacent pairs. Their loss objectives differ, so the plots use separate axes.

## Diagnostic method

| Item | Method |
| --- | --- |
| Episodes | 0, 13, 26, 40, 54, all from the training set |
| Observations | Approximately 20%, 50%, 70% through each episode; 15 total |
| Inputs | Current two-view images and six-joint state; SmolVLA also receives task text |
| Reset / seed | Policy reset per observation; seed `42 + probe index` |
| Horizon | First 50 ACT steps out of 100; all 50 SmolVLA steps |
| nMAE | Mean absolute error divided by each joint's training-action standard deviation |
| nRMSE | Root mean squared error after the same scaling |
| R² | Pool 15×50 targets per joint, then average the six joint scores |

The hold-state baseline repeats the current position for 50 steps. Tolerance hit rates use a fraction of the same training-action standard deviation. These are action-fitting diagnostics on seen observations; they do not measure grasp success or generalization. No independent split was used.

The action plot uses the middle saved observation: episode26, halfway through the episode. It shows predicted targets, separate from the measured positions in the video. Position units follow the calibrated servo convention.

## Deployment recordings

The deployment recordings use SmolVLA on the SO-101. Their source directories carry the checkpoint identifier `050000`. The table records the visible behavior and recording completeness; selected clips are separate from aggregate success statistics.

| Trial | Data rows | Fixed / wrist frames | Finalized episodes | Observation |
| --- | ---: | ---: | ---: | --- |
| 02 | 375 | 100 / missing | 0 | Truncated fixed video |
| 03 | 367 | 367 / 367 | 1 | Box tips; no placement; hand enters near end |
| 04 | 379 | 379 / 379 | 1 | Grasp/place visible; cable handling and human reset |
| 05 | 376 | 376 / 376 | 1 | No placement within clip; hand/cable interaction |
| 06 | 563 | 563 / 426 | 0 | Fixed view shows grasp/place; wrist ends early |
| 07 | 563 | 563 / 115 | 0 | Box tips; wrist ends early |
| 08 | — | — / — | 0 | No saved episode |

Trial06 supplies the homepage recording. Training episode54 is human teleoperation. Demonstration replay resends recorded targets; scripted control uses handwritten targets. Neither category is presented as model inference. Directory numbers and selected clips cannot establish a complete autonomous attempt count.

| Physical evaluation | Result |
| --- | --- |
| Verified autonomous successes / attempts | — / — |
| Success rate / completion-time distribution | — |
| Historical success threshold | Not recorded |
| Held-out positions or environments | Not recorded |

For a new evaluation, fix the checkpoint hash, calibration, camera mapping, start/object positions, time limit and clipping settings. Define success as grasping, transporting, releasing and leaving the box inside the target without intervention, with explicit dwell/timeout thresholds. Record all attempts, failures, interventions and aborted saves in [physical_trials.csv](../../results/so101/physical_trials.csv). Use the same conditions for both models.

<a id="video-and-joint-motion"></a>
## Video and joint motion

Trial06 pairs its fixed/wrist videos with 563 rows from `data/chunk-000/episode_000000.parquet`. Solid teal is measured position; dashed amber is the sent target. Joint order is shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper. With `use_degrees=false`, body joints use −100 to 100 and the gripper 0 to 100. The stored values, including a slightly negative gripper target, are preserved. No motor current or torque was recorded.

The two views follow the saved frame indexes. Fixed-video timestamps match the data within 0.90 microseconds and wrist timestamps within 0.45 microseconds, reflecting float32 rounding. These checks establish file alignment, not exposure synchronization or camera-to-controller latency.

The fixed view contains 563 frames (18.77 s); the wrist view contains 426 (14.20 s). An ending card marks the missing final wrist interval. Both composite formats keep the full fixed-camera interval at 1× with no time cuts. MP4 is 1440×900 at 30 FPS; GIF is 960×600 at 10 FPS. The CSV and [source record](../../results/so101/telemetry/trial06.json) retain samples, units, joint order and alignment hashes.

## Media files

| Files in `media/so101/` | Contents |
| --- | --- |
| `deployment-trial06-telemetry.*` | Two views and synchronized joint traces |
| `deployment-trial06.mp4` / `.gif` | Full fixed-camera recording; native MP4 and smaller preview |
| `deployment-trial06-display-1280.mp4` | 1280×960 display copy, Lanczos resize and mild sharpening |
| `deployment-trial06-two-view.*` | Equal-size views, with the missing wrist interval marked |
| `deployment-trial04-two-view.*` | Full 12.63 s; cable handling and reset included |
| `deployment-trial03-two-view.*` | Full 12.23 s; box tips before placement |
| `teleoperation-episode54.*` | Full 12.23 s human demonstration |
| `task-sequence.webp` | Trial06 fixed frames 241, 321, 401, 482 |
| `camera-inputs.webp` | Both views at frame210 of demonstration54 |

All videos play at 1× without time cuts. Two-view MP4s are 1280×600 at 30 FPS; their GIFs are 768×360 at 10 FPS. Camera orientation and field of view stay unchanged. Display upscaling cannot recover details lost to focus or compression. The native 640×480 video remains available.

The [media manifest](../../results/so101/media_manifest.json) records source SHA256 hashes, frame counts and processing. Source times are preserved file modification times, not authenticated capture dates. Selected MP4s use H.264 and ordinary Git storage; GIFs provide GitHub previews.

Rebuild from the original local dataset directories after installing the tools extra:

```bash
python scripts/so101/build_showcase.py --dataset-root ../datasets
python scripts/so101/build_telemetry_showcase.py --dataset-root ../datasets
```

The scripts read the existing demonstrations and trial03/04/06, write display assets and update the manifest. Original recordings stay unchanged.
