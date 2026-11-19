# Demonstration sources and video processing

## Homepage recording and two-view layout

The lead clip is `eval_medicine_box_black_050000_trial06/videos/chunk-000/observation.images.fixed/episode_000000.mp4`: 563 frames, 30 FPS, 640×480, AV1, 18.7667 seconds. Its fixed view shows the box lifted and released into the black target. No human intervention is visible in that camera's full interval; that visual observation alone cannot prove which controller generated the actions.

The source was supplied among deployment recordings. The `050000` name is retained, but the model family and exact checkpoint hash are absent. Its preserved modification time is September 29, 2026, 17:52:20 in the supplied local filesystem. This predates the recorded October 7 SmolVLA training run and is consistent with ACT, without proving it.

The companion handeye video contains only 426 frames (14.20 seconds). The two-view export shows these real frames alongside the fixed view, then shows a **HANDEYE RECORDING ENDED** text card for the final 137 fixed-camera frames (4.57 seconds). It never freezes the last camera frame, loops footage, or generates a replacement view. The source dataset metadata reports zero completed episodes, even though a full fixed video and 563 Parquet rows survive. These artifacts are not silently repaired or counted as a complete evaluated episode.

The two views are paired by frame index within the same saved episode. This represents the recorded dataset alignment, not measured hardware exposure synchronization. Title, camera names, source type, playback speed and elapsed time appear outside the 640×480 camera images. Camera orientation and field of view are preserved.

## Published assets

| Asset | Transformations |
| --- | --- |
| `deployment-trial06-telemetry.mp4` / `.gif` | Fixed camera, smaller wrist view and six progressively revealed measured/sent-position traces; 563 samples; full 18.77 s at 1×; explicit missing-wrist card |
| `deployment-trial06.mp4` | Full native 640×480 interval, AV1 to H.264, 30 FPS, no audio, no crop or time cut |
| `deployment-trial06.gif` | Same full interval and field of view, 480×360 / 10 FPS GIF preview, 1× playback |
| `deployment-trial06-display-1280.mp4` | Same full interval, Lanczos resize to 1280×960 and mild unsharp filter; 1× playback |
| `deployment-trial06-two-view.mp4` / `.gif` | Fixed and handeye side by side; full 563-frame fixed interval, explicit card after handeye frame426; labels outside the images; 1× |
| `deployment-trial04-two-view.mp4` / `.gif` | Both complete views at native 640×480 placed side by side with labels; full 379-frame interval, 1×; human cable handling and reset retained |
| `teleoperation-episode54.mp4` / `.gif` | Both views of training episode54 with labels, side by side; full 367-frame interval, 1×; human demonstration |
| `deployment-trial03-two-view.mp4` / `.gif` | Both complete views with labels; full 367-frame interval, 1×; box tips, no placement, human hand at end |
| `task-sequence.webp` | Selected fixed-camera frames 241, 321, 401, 482 of trial06; resized storyboard, source times shown; not a continuous video |
| `camera-inputs.webp` | Fixed and handeye frame 210 from demonstration54; native camera images with labels outside them |

The labeled two-view MP4s are 1280×600 at 30 FPS. GIF previews are 768×360 at 10 FPS, with the same interval and 1× playback. Their combined width is a layout choice, not additional source resolution.

The display copy is an **upscale of a 640×480 recording**. It cannot recover scene detail lost to focus, motion blur, exposure, or compression. No generated texture, AI restoration, object replacement, interpolation of robot motion, or altered grasp trajectory is used. The native copy remains the evidentiary reference.

The small selected videos under `media/so101/` are stored as regular Git files for direct GitHub download. Homepage MP4 links use `?raw=true` to open the download directly; GIFs provide the inline preview. GitHub's file page does not provide an embedded player for these MP4s. Upstream LFS rules for test fixtures are retained; full datasets and model weights are excluded from the source release.

`results/so101/media_manifest.json` contains original source SHA256 hashes, decoded frame counts, source times, published sizes, and processing commands. The source dataset files remain unchanged. Source camera keys and sampled original frames were inspected before selecting the clips.

<a id="video-and-joint-motion"></a>
## Video and joint motion

The homepage pairs trial06 camera footage with the same episode's `data/chunk-000/episode_000000.parquet`. Its 563 rows store `observation.state`, `action`, `frame_index` and `timestamp`. The six columns follow the metadata order: shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll and gripper. Solid teal shows measured position; dashed amber shows the recorded target returned by `robot.send_action`. Values and a shared time cursor advance with the camera frames. Only the history through the current sample is revealed.

The source has consecutive frame indices 0–562. All fixed-camera presentation timestamps agree with the corresponding Parquet timestamps within **0.90 microseconds**; the 426 wrist frames agree within **0.45 microseconds**. These tiny differences are float32 timestamp rounding, not a measured camera-to-controller latency. The recorded alignment does not establish hardware exposure synchronization or a measured 30 Hz end-to-end control rate. The missing wrist interval after 14.20 s remains an explicit card.

The robot uses `use_degrees=false`: body positions use the calibrated **−100 to 100** convention and the gripper uses **0 to 100**. The export preserves saved values, including a slightly negative sent gripper target; it does not clip, smooth or reinterpret them as degrees, radians, current or torque. There are no current or torque columns in the source Parquet. The plot is recorded telemetry, separate from the offline checkpoint-prediction plots in the results section.

The MP4 is 1440×936 / 30 FPS. The fixed image remains native 640×480, and the wrist image is reduced to 304×228 without cropping. The GIF is 960×624 / 10 FPS at 1× speed. The entire fixed-camera interval is retained without time cuts. This composite is a new publication asset made from the historical recording, not a new real-robot test or live stream. Exact controller and checkpoint identity remain unverified.

The selected [563-row CSV](../../results/so101/telemetry/trial06.csv) retains timestamps and both position vectors at nine significant digits, sufficient to round-trip the source float32 values. The [provenance record](../../results/so101/telemetry/trial06.json) includes source hashes, joint order, alignment checks, units and CSV hash. Full datasets remain excluded.

## Rebuild the presentation assets

```bash
python -m pip install -r requirements-so101-media.txt
python scripts/so101/build_showcase.py --dataset-root ../datasets
python scripts/so101/build_telemetry_showcase.py --dataset-root ../datasets
```

The script reads the existing `medicine_box_black_5_20s_v1` and deployment trial03/04/06 directories. It regenerates labeled two-view MP4/GIF files and the two real-frame illustrations, then updates their manifest entries. It does not modify source data. Use `python scripts/so101/plot_results.py` with the analysis dependencies to regenerate the separate loss and action-chunk plots.

## Control labels

| Category | What it means here |
| --- | --- |
| Human teleoperation | The leader provides follower targets; episode54 is labeled this way |
| Demonstration replay | Previously recorded targets are sent again; no such video is showcased as inference |
| Scripted control | Handwritten targets or trajectories; no such video is showcased as inference |
| Model autonomous execution | Targets come from a specified learned checkpoint without operator intervention; exact per-trial identifiers are missing for the surviving deployment clips |

The homepage therefore uses **deployment recording**, with the unresolved provenance stated beside the video. It does not upgrade uncertain footage to a verified autonomous benchmark.
