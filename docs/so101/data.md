# Demonstrations and data

## Recorded dataset

`medicine_box_black_5_20s_v1` contains 55 episodes, 20,512 time steps, and 110 camera videos. The only task is `Pick up the green medicine box and place it in the black target area`. The metadata declares `train: 0:55`, so no held-out split exists for these trained checkpoints.

| Field | Stored representation |
| --- | --- |
| `observation.images.fixed` | 640×480 RGB AV1 video, 30 FPS |
| `observation.images.handeye` | 640×480 RGB AV1 video, 30 FPS |
| `observation.state` | `float32[6]` |
| `action` | `float32[6]`, actual sent action |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | Per-step alignment and task lookup |

Joint order: `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos`. The archived SO-101 code uses `use_degrees=false`: arm joints use the signed normalized servo range, with the gripper in its own normalized range. The dataset metadata does not attach physical units. Do not relabel these values as radians or apply an arbitrary degrees conversion.

```text
medicine_box_black_5_20s_v1/
├── meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl}
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/observation.images.{fixed,handeye}/episode_000000.mp4
```

The source format is already LeRobot v2.1. There is no required v3 conversion or separate raw-to-LeRobot exporter for this experiment.

## Record a new dataset

From the repository root, use a new dataset name and output directory. Keep the camera keys unchanged. Bash example:

```bash
python -m lerobot.record \
  --robot.type=so101_follower --robot.port=FOLLOWER_PORT --robot.id=my_awesome_follower_arm \
  --robot.cameras='{fixed: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, handeye: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}' \
  --teleop.type=so101_leader --teleop.port=LEADER_PORT --teleop.id=my_awesome_leader_arm \
  --dataset.repo_id=local/medicine_box_new --dataset.root=../datasets/medicine_box_new \
  --dataset.single_task="Pick up the green medicine box and place it in the black target area" \
  --dataset.fps=30 --dataset.num_episodes=5 --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 --dataset.push_to_hub=false --display_data=false
```

PowerShell accepts the same arguments on a single line; replace Bash's trailing `\` with PowerShell's backtick if splitting lines. Use the observed device indexes, not the example 0/1 mapping. `episode_time_s=20` is a new recording limit, not a claim about the old dataset's measured duration. Interactive early exit can shorten episodes.

Recording reads the observation, reads the leader's action, sends it to the follower, then stores the returned action together with the observation. The two cameras are read in the same dataset control cycle; no hardware trigger synchronization is established.

## Readback and processing

```bash
python scripts/so101/audit_dataset.py --root ../datasets/medicine_box_black_5_20s_v1 --decode-videos
python -m lerobot.replay --robot.type=so101_follower --robot.port=FOLLOWER_PORT --robot.id=my_awesome_follower_arm --dataset.repo_id=local/medicine_box_black_5_20s_v1 --dataset.root=../datasets/medicine_box_black_5_20s_v1 --dataset.episode=54
```

The audit opens data read-only, checks six-value state/action arrays, finite values, timestamps, contiguous frame indexes, metadata lengths, and every decoded video frame count. Replay sends the recorded actions to the robot; label resulting video as **demonstration replay**, not autonomous execution.

The supplied dataset audit passed all 55 episodes and 110 videos. An existing `outputs/dataset_repairs/` backup records an earlier camera-swap repair for episode0. Its operation log and training-time chronology are incomplete. The release preserves the current dataset and validates dimensions and counts; this does not establish camera semantics at the exact time each historical model trained. Recheck the two views visually before retraining. Do not run a new swap or overwrite a source dataset to make the display look better.

Dataset normalization is aggregated by the existing LeRobot loader from per-episode statistics. Fresh training creates matching policy normalization buffers; inference uses those saved in `model.safetensors`. A newly collected dataset's statistics must not replace an old checkpoint's buffers.

## Split policy

The archived results are training diagnostics. A future generalization study should assign whole episodes to train/validation/test before retraining, keep evaluation episodes out of training, and save that split in the run configuration. Selecting a few frames from already-used episodes does not create an independent test set.
