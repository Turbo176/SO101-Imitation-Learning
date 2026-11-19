# Setup and data collection

Use this checkout of LeRobot 0.3.4 with Python 3.10. The SO-101 changes and dataset format depend on this version.

## Install

Activate your environment, then run from the repository root:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[feetech,smolvla]" -c requirements.txt
python -m pip check
python scripts/so101/verify_results.py
```

Choose a PyTorch wheel supported by your driver; use the CPU index for offline work on a CPU host. `requirements.txt` pins the model compatibility packages checked in the Windows deployment environment. Core dependencies and optional extras are in `pyproject.toml`. The original Linux training environment was not fully locked.

For plots and video exports, add the optional tools:

```bash
python -m pip install -e ".[tools]" -c requirements.txt
```

## Ports and cameras

```bash
python -m lerobot.find_port
python -m lerobot.find_cameras opencv
```

Follow the port tool's unplug/replug prompt to identify each arm. Use the saved camera images to assign `fixed` to the workspace and `handeye` to the gripper. Camera indexes can change after reconnecting USB devices.

The experiment used two 640×480 RGB streams at 30 FPS. Camera models and exposure settings were not saved. The OpenCV config accepts `fourcc: "MJPG"` when the camera needs MJPEG for the requested mode.

## Motors and calibration

Run motor setup only for a newly assembled or readdressed arm. Keep the existing IDs and calibration for a previously configured robot.

```bash
python -m lerobot.setup_motors --robot.type=so101_follower --robot.port=FOLLOWER_PORT
python -m lerobot.setup_motors --teleop.type=so101_leader --teleop.port=LEADER_PORT

python -m lerobot.calibrate --robot.type=so101_follower --robot.port=FOLLOWER_PORT --robot.id=my_awesome_follower_arm
python -m lerobot.calibrate --teleop.type=so101_leader --teleop.port=LEADER_PORT --teleop.id=my_awesome_leader_arm
```

Replace the port placeholders with your discovered ports. The robot ID selects its calibration file under `$HF_LEROBOT_CALIBRATION/robots/so101_follower/`; the leader uses `teleoperators/so101_leader/`. The default root is the Hugging Face cache's `lerobot/calibration` directory. Override it with `--robot.calibration_dir` or `--teleop.calibration_dir`.

Calibration converts servo counts to joint positions. It is separate from dataset mean/std statistics and the checkpoint's saved normalization buffers.

Check manual leader–follower control before recording:

```bash
python -m lerobot.teleoperate --robot.type=so101_follower --robot.port=FOLLOWER_PORT --robot.id=my_awesome_follower_arm --teleop.type=so101_leader --teleop.port=LEADER_PORT --teleop.id=my_awesome_leader_arm
```

<a id="data-collection"></a>
## Record demonstrations

Use a new dataset name and the camera indexes found above. This command uses Bash line continuations; in PowerShell, put it on one line or use backticks.

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

Each control iteration stores the observation and the action returned by `robot.send_action`, including any applied clipping. The cameras share dataset timestamps; hardware exposure synchronization was not measured.

## Dataset format and readback

`medicine_box_black_5_20s_v1` is a LeRobot v2.1 dataset: 55 episodes, 20,512 rows and 110 videos. All episodes belong to `train: 0:55`; there is no held-out split. Actual episode lengths are 9.30–13.37 seconds despite the `20s` in the directory name.

```text
medicine_box_black_5_20s_v1/
├── meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl}
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/observation.images.{fixed,handeye}/episode_000000.mp4
```

| Field | Format |
| --- | --- |
| `observation.images.fixed`, `observation.images.handeye` | RGB AV1 video, 640×480, 30 FPS |
| `observation.state`, `action` | Six float32 joint positions |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | Alignment and task lookup |

Joint order is shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper. With `use_degrees=false`, arm positions use the calibrated −100 to 100 convention and the gripper uses 0 to 100. These are normalized servo positions, not radians or degrees.

```bash
python scripts/so101/audit_dataset.py --root ../datasets/medicine_box_black_5_20s_v1 --decode-videos
python -m lerobot.replay --robot.type=so101_follower --robot.port=FOLLOWER_PORT --robot.id=my_awesome_follower_arm --dataset.repo_id=local/medicine_box_black_5_20s_v1 --dataset.root=../datasets/medicine_box_black_5_20s_v1 --dataset.episode=54
```

The audit reads the data without modifying it and checks shapes, finite values, timestamps, indexes and decoded video counts. Replay sends recorded targets to the arm. The dataset already uses v2.1 and needs no format conversion.

An old local backup records a camera-swap repair for episode0; its timing relative to training is incomplete. The current dataset passes the audit, but inspect both views before retraining. Keep original data and repair records separately. For a new generalization study, split whole episodes before training.

## Common problems

| Symptom | Check |
| --- | --- |
| Reversed or missing views | Probe cameras again, close other camera apps, check `fixed`/`handeye` assignment and supported capture modes |
| Serial read/write failures | Port, power, motor IDs, cables and USB hub load; SO-101 read/startup retries default to 3 |
| Unexpected action scale | Robot ID, physical calibration, joint order, `use_degrees=false` and saved model normalization |
| AV1 / TorchCodec error | Use `dataset.video_backend=pyav`, as in the provided training recipes |
| Empty or partial episode | Check metadata, Parquet and decoded videos together; an interrupted save may leave files without a finalized episode |
