# Deployment

<a id="configuration-contract"></a>
## Robot and model settings

| Item | Required setting |
| --- | --- |
| Cameras | `fixed`: workspace; `handeye`: gripper |
| Images | 640×480 RGB; HWC uint8 from cameras, CHW float32 `[0,1]` at policy input |
| Joint order | Shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper |
| Calibration | The physical follower's calibration, `use_degrees=false` |
| Normalization | Buffers from the loaded `model.safetensors` |
| Model directory | Matching `config.json` and complete trained weights |
| SmolVLA task | `Pick up the green medicine box and place it in the black target area` |

Calibration converts servo counts to positions. The dataset supplies training statistics; the checkpoint stores the corresponding normalization buffers. Keep all three consistent. New dataset statistics should not replace an old checkpoint's buffers.

## Offline check

Fetch the small SmolVLA processor/config files, then check an existing trained checkpoint:

```bash
python scripts/so101/prepare_assets.py --output assets/smolvlm
python scripts/so101/check_model.py \
  --model outputs/smolvla/050000/pretrained_model --assets assets/smolvlm \
  --dataset ../datasets/medicine_box_black_5_20s_v1 --episode 25 --frame 60 \
  --device cuda --output outputs/offline_check.json
```

The assets come from `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`. The helper excludes backbone weights. Use your actual model path, including an imported server checkpoint if applicable.

The check strictly loads the trained weights in FP32, validates normalization and checks finite `[50,6]` chunks and `[1,6]` queued actions. It opens no robot or camera connection. Omit `--dataset` for synthetic input, or select `--device cpu` for a CPU check.

## SmolVLA live runner

Probe the cameras:

```bash
python scripts/so101/run_policy.py --list-cameras --snapshot-dir outputs/camera_probe
```

Check live observations and predictions before sending policy targets:

```bash
python scripts/so101/run_policy.py --mode live \
  --robot-port FOLLOWER_PORT --robot-id my_awesome_follower_arm \
  --camera-fixed 0 --camera-handeye 1 \
  --model outputs/smolvla/050000/pretrained_model --assets assets/smolvlm \
  --seconds 20 --max-relative-target 5 \
  --snapshot-dir outputs/live_frames --output outputs/live_dry.json
```

Live mode connects and configures motors, including torque, even without `--execute`. Add `--execute` to send policy targets:

```bash
python scripts/so101/run_policy.py --mode live --execute \
  --robot-port FOLLOWER_PORT --robot-id my_awesome_follower_arm \
  --camera-fixed 0 --camera-handeye 1 \
  --model outputs/smolvla/050000/pretrained_model --assets assets/smolvlm \
  --seconds 20 --max-relative-target 5 --output outputs/live_execute.json
```

Use your discovered ports and camera indexes. The target limit uses normalized joint units and changes the executed trajectory; record it in evaluations. Normal exit and `Ctrl+C` disconnect the robot. Keep the physical power switch accessible.

The runner computes each chunk synchronously, then consumes queued actions. Motion can pause while the next chunk is generated. A historical GTX 1650 Ti offline check measured about 0.75 s per warm 50-step chunk on one observation, excluding camera/serial work. The 30 FPS target is not a measured end-to-end control rate. The current runner fixes duplicate chunk computation and camera-key mapping; those changes have offline regression coverage but have not been rerun on the arm.

## Record a rollout

Use the standard recorder to save camera images, measured positions and sent targets. For ACT, omit the teleoperator:

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

For SmolVLA, use its model directory and add `--policy.vlm_model_name=assets/smolvlm --policy.load_vlm_weights=false`. Choose a new dataset name/root. Save the command, checkpoint hash, calibration ID, camera mapping, full video and outcome for each attempt. Historical hashes are in [checkpoint_manifest.json](../../results/so101/checkpoint_manifest.json).

## Common problems

| Symptom | Check |
| --- | --- |
| Missing offline assets | Run `prepare_assets.py`; keep the processor files and full trained `model.safetensors` locally |
| Wrong viewpoint or action scale | Recheck camera identities, robot calibration and model normalization |
| Pauses at chunk boundaries | Measure generation time and the full control loop; the synchronous runner waits for inference |
| Partial recording | Audit metadata and both decoded videos before treating it as a complete episode |
