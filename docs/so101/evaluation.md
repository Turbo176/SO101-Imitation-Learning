# Results and evaluation scope

## Evidence included in the repository

The loss CSVs are retained experiment exports. ACT logs contain rounded terminal values at a verified 200-step cadence. The SmolVLA CSV contains full-precision 100-step windows; the 200-step display file averages adjacent pairs. The original ACT displayed-step rounding is preserved in its own column.

`checkpoint_predictions.npz` is a compact result artifact, not model weights: ACT and SmolVLA predictions for 5 checkpoints × 15 matched observations × 50 steps × 6 joints, plus targets, a hold-state baseline, and normalization scales. `scripts/so101/verify_results.py` recomputes every published checkpoint metric. It currently reproduces the original CSV with zero numerical difference.

```bash
python scripts/so101/verify_results.py
python -m pip install -r requirements-so101-analysis.txt
python scripts/so101/plot_results.py
```

The plotting script also writes `media/so101/action-chunks.svg` and its PNG copy. It uses the middle of the 15 stored observations (zero-based index7, observation8), rather than selecting the best-fitting example: episode26 at the recorded 0.5 fraction. The two 50k predictions and the recorded targets are drawn for all six joints over the same 50-step horizon. These are offline predicted chunks, not measured executed trajectories. The y-axis uses recorded position units (`use_degrees=false`, normalized servo-position convention).

## Diagnostic protocol

- Episodes: 0, 13, 26, 40, 54 from the **training set**.
- Observations: approximately 20%, 50%, 70% through each episode, 15 total.
- Inputs: current two-view images and six-joint state; SmolVLA also uses the recorded task string. Recorded future actions are targets only.
- Each observation starts with a policy reset. The original evaluation fixed the random seed to `42 + probe index`.
- ACT's native 100-step prediction is truncated to the first 50 steps, matching SmolVLA's native 50-step output.
- nMAE is the mean of `abs(prediction - target) / action_std[joint]`. nRMSE is the root mean squared error after the same scaling. Both models use the same training-action standard deviations.
- R² pools 15×50 targets per joint, computes one value for each joint, then takes their unweighted mean. It is not an accuracy percentage.
- Tolerance hit rates use an explicit fraction of the joint's training standard deviation. They are not task-success measurements. Adjacent actions are correlated and cannot be counted as independent attempts.

The hold-state baseline repeats the current six-joint observation for 50 steps. Its arrays and recomputable metrics are included to make short-horizon fitting comparisons meaningful. These diagnostics do not establish generalization or the best physical checkpoint.

## Surviving deployment artifacts

The names `trial02` through `trial08` do not establish a consecutive, complete evaluation log. Outcomes below describe visible evidence, not adjudicated autonomous successes. All source timestamps are preserved filesystem modification times in the supplied copy; they are supporting evidence rather than authenticated video creation times.

| Artifact | Parquet rows | Fixed / handeye frames | Metadata episodes | Visible evidence / limitation |
| --- | ---: | ---: | ---: | --- |
| trial02 | 375 | 100 / missing | 0 | Truncated fixed video; incomplete save |
| trial03 | 367 | 367 / 367 | 1 | Box tips over; no target placement; human hand enters near end |
| trial04 | 379 | 379 / 379 | 1 | Grasp and placement visible; cable handling and later human reset visible |
| trial05 | 376 | 376 / 376 | 1 | No target placement within clip; human hand/cable interaction visible |
| trial06 | 563 | 563 / 426 | 0 | Fixed video shows grasp and placement; handeye ends early; metadata not finalized |
| trial07 | 563 | 563 / 115 | 0 | Box tips; no placement within surviving fixed clip; handeye ends early |
| trial08 | missing | missing / missing | 0 | No saved episode |

The `050000` directory suffix suggests the checkpoint step, and the September 29 video timestamps align with the retained ACT training record. However, no per-trial command, model path, hash, or explicit control-mode flag survives. Policy identity remains unverified. Do not infer the architecture from arm motion or label all seven directories as completed attempts.

The homepage's trial06 is the clearest surviving single-view pick-and-place recording. Its full interval is presented at 1×, with no removal of failure segments or waiting time. The two-camera trial04 and the human demonstration are separate examples. See [media provenance](media.md).

## Physical outcome fields

| Metric | Value |
| --- | --- |
| Verified autonomous successful attempts | — |
| Total autonomous attempts under one protocol | — |
| Complete-task success rate | — |
| Completion-time distribution | — |
| Historical success/failure threshold | — |
| Held-out object positions or environments | — |

Leave these values blank until the controller source and complete attempt log are available, or run a new evaluation. The publication preparation does not run the robot to fill missing historical results.

## Protocol for a new evaluation

Fix the checkpoint hash, camera mapping, calibration ID, robot convention, start-pose distribution, object pose, black target boundary, time limit, and target clipping before collecting attempts. Define success as lifting the box, transporting it into the target, releasing it, and leaving it inside the boundary without human assistance; record the exact dwell/timeout thresholds selected for the new run.

Save every attempted episode, including failures, with full video and an outcome. Record interventions and aborted or corrupted saves explicitly. Report `successful / attempted` with the chosen rule for aborts, and break down failures such as missed grasp, dropped object, outside-target placement, timeout, intervention, or recording failure. Compare ACT and SmolVLA under the same conditions. Keep selected demonstration clips separate from this complete trial sheet.

`results/so101/physical_trials.csv` inventories the surviving artifacts and leaves controller, checkpoint hash, adjudicated outcome, and success fields empty. It is a starting record, not a completed evaluation.
