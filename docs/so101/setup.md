# Setup and calibration

This checkout is LeRobot 0.3.4 at upstream commit `882c80d446a63a44868c67ae535467af32ce0e80`, with the local SO-101 changes listed in [provenance](provenance.md). Keep this checkout; installing a current LeRobot release would change APIs and the dataset format assumptions.

## Python environment

The recorded inference environment uses Python 3.10, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, Transformers 4.51.3, and Diffusers 0.33.1. Reuse an existing compatible environment when available.

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[feetech,smolvla]" -c requirements-so101.txt
python -m pip check
python -m lerobot.find_cameras --help
python scripts/so101/verify_results.py
```

For a CPU host, select the official CPU wheel index for the first command and use `--device cpu` for offline tools. Training and real-time throughput on CPU were not established. The CUDA wheel is specific to a compatible NVIDIA host and driver.

`requirements-so101.txt` constrains the model compatibility packages from the existing deployment environment. The remaining package ranges come from `pyproject.toml`; it is not a complete lock for the original Linux training environment. Use `pyav` for the portable dataset recipes. System FFmpeg is needed for video processing; PyAV and the installed imageio-ffmpeg binary can also inspect and transcode these recordings.

## Connect the robot and cameras

```bash
python -m lerobot.find_port
python -m lerobot.find_cameras opencv
```

Identify the leader and follower ports by unplugging one adapter at a time when prompted. Windows uses COM port names; Linux uses serial device paths. Do not copy a port name from a different machine. Camera numbering can change when USB devices reconnect. Inspect the saved camera images to assign `fixed` to the whole workspace and `handeye` to the gripper view.

The experiment records 640×480 RGB at 30 FPS. The local OpenCV config accepts `fourcc: "MJPG"` for cameras that require MJPEG to expose a requested mode; it does not change already-recorded source resolution. Camera model numbers, lens settings, exposure values, and exact USB topology were not saved.

## Motor setup and calibration

Motor setup is for a newly assembled or readdressed arm. Reuse existing motor IDs and calibration files for the experiment robot.

```bash
python -m lerobot.setup_motors --robot.type=so101_follower --robot.port=FOLLOWER_PORT
python -m lerobot.setup_motors --teleop.type=so101_leader --teleop.port=LEADER_PORT

python -m lerobot.calibrate --robot.type=so101_follower --robot.port=FOLLOWER_PORT --robot.id=my_awesome_follower_arm
python -m lerobot.calibrate --teleop.type=so101_leader --teleop.port=LEADER_PORT --teleop.id=my_awesome_leader_arm
```

Replace `FOLLOWER_PORT` and `LEADER_PORT` with the discovered values. For existing checkpoints, keep the robot ID that selects the matching calibration. Calibration defaults to `$HF_LEROBOT_CALIBRATION/robots/so101_follower/<robot-id>.json` and the equivalent `teleoperators/so101_leader/` directory; without an override, the root is the Hugging Face cache's `lerobot/calibration` directory. The CLI also accepts `--robot.calibration_dir=...` and `--teleop.calibration_dir=...` for a local directory.

Calibration maps servo counts into the policy's joint-position convention. It is separate from dataset mean/std normalization. A different robot requires its own physical calibration; neither the model weights nor the demonstration statistics calibrate the hardware.

## Connection check

```bash
python -m lerobot.teleoperate --robot.type=so101_follower --robot.port=FOLLOWER_PORT --robot.id=my_awesome_follower_arm --teleop.type=so101_leader --teleop.port=LEADER_PORT --teleop.id=my_awesome_leader_arm
```

This is manual leader–follower control. It is not policy inference. Run the camera-enabled collection command in the [data guide](data.md) after confirming joint direction and camera assignment.
