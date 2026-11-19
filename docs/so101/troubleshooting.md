# Troubleshooting

## Cameras reversed or unavailable

Run `python -m lerobot.find_cameras opencv`, inspect the images, and set the actual indexes. The trained keys are `fixed` and `handeye`; swapping those views preserves tensor shape but changes the learned observation semantics. Close applications that hold the cameras. Recheck indexes after reconnecting USB devices.

The local camera implementation supports `fourcc: "MJPG"` and applies it before the requested size/rate. Use it only for a device that supports this capture mode. A larger negotiated frame does not improve focus or recover missing detail in old footage. Historical video evidence is 640×480 regardless of higher-resolution camera capabilities.

## Serial read or startup write failures

Verify the follower port, power, motor IDs, and cable seating. The local SO-101 config has `position_read_num_retry=3` and `configuration_num_retry=3`; these are additional retries, not a cure for disconnected hardware. The local leader has analogous startup support. Sharing a USB hub with cameras can expose intermittent serial errors. Avoid changing calibration to compensate for communication failures.

## Wrong calibration or action scale

The robot ID selects its calibration file. Use the same joint ordering and `use_degrees=false` convention as the checkpoint. Preserve the saved mean/std normalization buffers. The relative target limit is in the robot's recorded normalized joint units, including the gripper's different range; it is not a degrees-per-step parameter.

## Missing offline assets

Run `python scripts/so101/prepare_assets.py --output assets/smolvlm` while online. The offline runners intentionally disable Hub access. They need the processor/config/tokenizer files and the complete trained checkpoint. `load_vlm_weights=false` prevents downloading an unrelated base-weight copy during inference; it does not remove the requirement for the trained `model.safetensors`.

## AV1 or torchcodec errors

The historical source videos are AV1. The checked local environment decodes them with PyAV. The portable training configs use `dataset.video_backend=pyav`; the archived historical configs used `torchcodec`. Select the backend explicitly for the installed platform. Converting showcase videos to H.264 improves browser playback but should not rewrite the training dataset.

## Empty or partially saved episodes

An early keyboard exit before the first frame can create an empty episode. The local recorder skips an empty episode buffer. A process interruption during video encoding may still leave Parquet, short videos, and unfinalized metadata, as seen in several historical deployment directories. Keep those artifacts separate from valid episodes; do not count a folder name as a successful attempt.

## Loss and checkpoint interpretation

Read the actual step from `training_state/training_step.json`. The `steps` field in a saved train config is the planned endpoint. ACT's L1-plus-KL loss and SmolVLA's flow-matching loss use different objectives. Their absolute values cannot rank physical policies. R² and tolerance hits from training observations are not robot success rates.

## Slow or interrupted live motion

The migrated runner generates one chunk when `select_action` needs to refill its queue. It blocks during that generation, then reads cached actions. Offline warm inference time excludes camera I/O, serial work, and the visible pause at each refill. Check actual rollout timing; the 30 FPS target is not a validated continuous 30 Hz controller. The runner's dry mode still connects and configures the arm, including torque.
