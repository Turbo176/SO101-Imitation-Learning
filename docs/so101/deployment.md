# Deploy on the SO-101

## Configuration contract

| Item | Must agree with the experiment |
| --- | --- |
| Camera keys and identity | `fixed` is the workspace; `handeye` is the gripper view |
| Image arrays | RGB HWC uint8 from cameras; CHW float32 `[0,1]` at policy input; 640×480 source |
| Joint order | shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper |
| Calibration | Physical follower's calibration and `use_degrees=false` convention |
| Normalization | The buffers belonging to the loaded `model.safetensors` |
| Model directory | Matching `config.json` and full trained weights; `train_config.json` for resume |
| SmolVLA language | `Pick up the green medicine box and place it in the black target area` |
| Recorded data rate | 30 FPS; this is a requested control target, not an end-to-end throughput measurement |

Calibration maps servo counts to joint values. Dataset statistics normalize those values for learning. Checkpoint buffers store the resulting normalization. These are three different artifacts; replacing one does not repair a mismatch in another.

## Offline checkpoint check

Keep the existing trained model at `outputs/smolvla/050000/pretrained_model`, or supply its actual path. Fetch only the fixed processor/config files:

```bash
python scripts/so101/prepare_assets.py --output assets/smolvlm
python scripts/so101/check_model.py --model outputs/smolvla/050000/pretrained_model --assets assets/smolvlm --dataset ../datasets/medicine_box_black_5_20s_v1 --episode 25 --frame 60 --device cuda --output outputs/offline_check.json
```

Processor source: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`. The helper downloads JSON/text assets, not model weights. The offline check loads the complete trained weights with `strict=True`, sets `load_vlm_weights=false`, checks the normalization buffers, and verifies finite `[50,6]` chunks and queued `[1,6]` actions. It does not connect the robot or cameras. Omit `--dataset` to use a synthetic compatibility input instead; synthetic input is not an evaluation result.

The migration preserves the existing `check_model.py` and `run_policy.py` implementation, replacing machine-specific defaults. Old deployment notes incorrectly described `"image" in "observation.images.fixed"` as false and claimed the recorder did not supply metadata. The current code matches `"image"` and calls `make_policy(..., ds_meta=dataset.meta)`. Those old explanations are not applicable to this checkout.

## Live SmolVLA runner

First identify the two camera indexes:

```bash
python scripts/so101/run_policy.py --list-cameras --snapshot-dir outputs/camera_probe
```

Then run live inference without sending policy targets:

```bash
python scripts/so101/run_policy.py --mode live --robot-port FOLLOWER_PORT --robot-id my_awesome_follower_arm --camera-fixed 0 --camera-handeye 1 --model outputs/smolvla/050000/pretrained_model --assets assets/smolvlm --seconds 20 --max-relative-target 5 --snapshot-dir outputs/live_frames --output outputs/live_dry.json
```

`robot.connect()` configures motors and enables torque even in this mode. No policy action targets are sent until `--execute` is supplied. Startup is not a hardware-free operation.

```bash
python scripts/so101/run_policy.py --mode live --execute --robot-port FOLLOWER_PORT --robot-id my_awesome_follower_arm --camera-fixed 0 --camera-handeye 1 --model outputs/smolvla/050000/pretrained_model --assets assets/smolvlm --seconds 20 --max-relative-target 5 --output outputs/live_execute.json
```

Use the discovered ports and indexes. The target clip is in recorded joint units, not degrees; it changes the executed trajectory and must be recorded when comparing outcomes. Normal exit and `Ctrl+C` disconnect the robot. Keep the workspace clear and use the physical power switch if motion must stop immediately.

The runner uses the policy's queued actions and generates each chunk once. It is synchronous: the robot waits while a new chunk is computed. The archived GTX 1650 Ti offline check measured about 0.75 seconds for a warm 50-step chunk on one recorded observation, excluding physical control. That does not prove continuous 30 Hz closed-loop control. This publication changes the runner's redundant chunk call; its revised control path has hardware-free regression coverage but has not been rerun on the arm.

## Record ACT or SmolVLA deployment

The existing LeRobot recorder supports policy inference and saves executed actions. Supply no teleoperator for a policy-only rollout. ACT example:

```bash
python -m lerobot.record \
  --robot.type=so101_follower --robot.port=FOLLOWER_PORT --robot.id=my_awesome_follower_arm \
  --robot.cameras='{fixed: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, handeye: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}' \
  --policy.path=outputs/checkpoints/050000/pretrained_model --policy.device=cuda \
  --dataset.repo_id=local/eval_medicine_box_act_50k_new --dataset.root=../datasets/eval_medicine_box_act_50k_new \
  --dataset.single_task="Pick up the green medicine box and place it in the black target area" \
  --dataset.fps=30 --dataset.num_episodes=1 --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 --dataset.push_to_hub=false --display_data=false
```

For SmolVLA, change the policy path to `outputs/smolvla/050000/pretrained_model`, provide `--policy.vlm_model_name=assets/smolvlm --policy.load_vlm_weights=false`, and use a new dataset name/root. With complete local weights, do not initialize the VLM backbone again. The recorded checkpoint hashes are in [checkpoint_manifest.json](../../results/so101/checkpoint_manifest.json).

For future evaluation, save the exact command/config, checkpoint hash, robot ID, camera mapping, clip limit, full video, and outcome for each attempt. Existing dataset metadata alone does not record policy identity.
