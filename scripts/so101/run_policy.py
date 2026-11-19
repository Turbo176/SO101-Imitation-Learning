"""Run the trained SmolVLA checkpoint locally (SO101 follower + 2 USB cameras).

Three modes, safest first:

1. offline synthetic  : no hardware at all, checks the checkpoint loads and emits actions.
2. live dry run       : connects arm + cameras, predicts real chunks, but sends nothing.
3. live control       : predicts with the real cameras and drives the arm.

The checkpoint is the one trained on `medicine_box_black_5_20s_v1`:
  state/action = 6 joints, cameras = observation.images.fixed + observation.images.handeye,
  task text   = "Pick up the green medicine box and place it in the black target area",
  action chunk = 50 steps, so one inference is followed by 49 cached steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
# Everything is already on disk: the checkpoint and the VLM assets folder.
# Blocking network access makes sure nothing silently re-downloads weights.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.dont_write_bytecode = True

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
DEFAULT_MODEL = ROOT / 'outputs/smolvla/050000/pretrained_model'
DEFAULT_ASSETS = ROOT / 'assets/smolvlm'
DEFAULT_TASK = 'Pick up the green medicine box and place it in the black target area'

FPS = 30
CHUNK = 50


def defaults() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='trained checkpoint directory')
    p.add_argument('--assets', type=Path, default=DEFAULT_ASSETS, help='local SmolVLM2 VLM assets folder')
    p.add_argument('--task', default=DEFAULT_TASK, help='language instruction; must match the training task')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--mode', choices=['offline', 'live'], default='offline')
    p.add_argument('--execute', action='store_true', help='live mode: actually move the arm (default is dry run)')
    p.add_argument('--seconds', type=float, default=20.0, help='live mode: length of the rollout')
    p.add_argument('--fps', type=float, default=FPS, help='control loop rate')
    p.add_argument('--robot-port', default=None, help='follower serial port; required in live mode')
    p.add_argument('--robot-id', default='my_awesome_follower_arm')
    p.add_argument('--camera-fixed', type=int, default=0, help='camera index for observation.images.fixed')
    p.add_argument('--camera-handeye', type=int, default=1, help='camera index for observation.images.handeye')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--camera-fps', type=int, default=FPS)
    p.add_argument('--max-relative-target', type=float, default=None,
                   help='safety clip in joint units per step (try 5 and raise it if the arm is too slow)')
    p.add_argument('--snapshot-dir', type=Path, default=None, help='write the first observation frames here')
    p.add_argument('--output', type=Path, default=None, help='write a JSON report here')
    p.add_argument('--list-cameras', action='store_true',
                   help='probe indices 0-4, save one frame each as a PNG, and exit (identify fixed/handeye)')
    return p.parse_args()


def probe_cameras(args) -> int:
    """Save one frame per camera index so the two viewpoints can be told apart."""
    import cv2

    out_dir = (args.snapshot_dir or BASE / 'camera_probe')
    out_dir.mkdir(parents=True, exist_ok=True)
    found = 0
    for index in range(5):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        frame = None
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            for _ in range(10):  # warm up: the first frames of a USB camera are often unusable
                ok, frame = capture.read()
            if not ok:
                frame = None
        capture.release()
        if frame is None:
            print(f'index {index}: no usable frame', flush=True)
            continue
        found += 1
        path = out_dir / f'probe_{index}.png'
        cv2.imwrite(str(path), frame)
        print(f'index {index}: {frame.shape[1]}x{frame.shape[0]} -> {path}', flush=True)
    if found == 0:
        print('no camera answered. Check the USB cables and close other programs using them.', flush=True)
        return 1
    print('\nOpen the PNGs: the gripper close-up is handeye, the whole workspace is fixed.', flush=True)
    print(f'Then run: --camera-fixed <index> --camera-handeye <index>', flush=True)
    return 0


# --------------------------------------------------------------------------------------
# checkpoint loading
# --------------------------------------------------------------------------------------
def load_policy(args):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    model, assets = args.model.resolve(), args.assets.resolve()
    required = [model / 'config.json', model / 'model.safetensors',
                assets / 'config.json', assets / 'tokenizer.json', assets / 'processor_config.json']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f'missing {path}')

    config = PreTrainedConfig.from_pretrained(model, local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise ValueError(f'{model} is not a SmolVLA checkpoint (got {type(config).__name__})')

    # Use the already-trained weights only; never pull the original VLM from the network.
    config.device = args.device
    config.vlm_model_name = str(assets)
    config.load_vlm_weights = False
    config.use_amp = False  # GTX 1650 Ti has no bf16; FP32 is the verified path.

    started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(model, config=config, strict=True, local_files_only=True)
    policy = policy.to(args.device).float().eval()
    load_seconds = time.perf_counter() - started

    info = {
        'model': str(model),
        'assets': str(assets),
        'device': args.device,
        'precision': 'float32',
        'chunk_size': config.chunk_size,
        'n_action_steps': config.n_action_steps,
        'input_features': sorted(config.input_features),
        'output_features': sorted(config.output_features),
        'load_seconds': round(load_seconds, 3),
    }
    step_file = model.parent / 'training_state/training_step.json'
    if step_file.is_file():
        info['checkpoint_training_step'] = json.loads(step_file.read_text(encoding='utf-8'))['step']
    return policy, config, info


# --------------------------------------------------------------------------------------
# observation -> policy batch
# --------------------------------------------------------------------------------------
def observation_to_batch(policy, config, observation: dict, task: str, device: str):
    """Convert a raw robot observation dict into the policy's expected batch."""
    import torch

    images = config.image_features
    batch: dict = {'task': task}
    for key in config.input_features:
        if key in images:
            # Dataset batches use fully qualified keys; SO101Follower exposes
            # raw camera names (fixed / handeye) in get_observation().
            source_key = key if key in observation else key.removeprefix('observation.images.')
            if source_key not in observation:
                raise KeyError(f'observation is missing camera "{key}"; got {sorted(observation)}')
            array = observation[source_key]
            expected = tuple(images[key].shape)  # (3, H, W) channel-first, as the policy stores it
            tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float().div(255).unsqueeze(0)
            if tuple(tensor.shape[1:]) != expected:
                raise ValueError(
                    f'camera "{key}" produced {tuple(array.shape)}, expected {(expected[1], expected[2], expected[0])} '
                    f'(height, width, channels); check --width/--height and the camera index'
                )
        elif key == 'observation.state':
            if key in observation:  # a ready-made 6-vector
                values = list(observation[key])
            else:  # raw SO101 reading: six "<motor>.pos" entries
                joint_keys = [f'{m}.pos' for m in
                              ('shoulder_pan', 'shoulder_lift', 'elbow_flex',
                               'wrist_flex', 'wrist_roll', 'gripper')]
                values = [observation[k] for k in joint_keys if k in observation]
            if len(values) != 6:
                raise KeyError(f'need all 6 joint positions, got {sorted(observation)}')
            tensor = torch.tensor([values], dtype=torch.float32)
        else:
            continue
        if not torch.isfinite(tensor).all():
            raise ValueError(f'non-finite value in "{key}"')
        batch[key] = tensor.to(device)
    return batch


def save_snapshot(directory: Path, batch: dict, config) -> None:
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    for key in config.image_features:
        if key not in batch:
            continue
        array = (batch[key][0].permute(1, 2, 0).cpu().numpy() * 255).round().astype('uint8')
        name = key.replace('observation.images.', '').replace('.', '_')
        path = directory / f'{name}.png'
        Image.fromarray(array, mode='RGB').save(path)
        print(f'  saved {path}', flush=True)


# --------------------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------------------
def run_offline(args, policy, config, result):
    import torch
    mean = policy.state_dict()['normalize_inputs.buffer_observation_state.mean']
    observation = {'observation.state': mean.tolist()}
    observation.update({
        key: np.zeros((feature.shape[1], feature.shape[2], feature.shape[0]), dtype=np.uint8)
        for key, feature in config.image_features.items()
    })
    batch = observation_to_batch(policy, config, observation, args.task, args.device)
    if args.snapshot_dir:
        save_snapshot(args.snapshot_dir, batch, config)
    with torch.inference_mode():
        policy.reset()
        started = time.perf_counter()
        chunk = policy.predict_action_chunk(dict(batch))
        if args.device == 'cuda':
            torch.cuda.synchronize()
        first = time.perf_counter() - started
        started = time.perf_counter()
        chunk = policy.predict_action_chunk(dict(batch))
        if args.device == 'cuda':
            torch.cuda.synchronize()
        warm = time.perf_counter() - started
    result.update(
        mode='offline',
        action_shape=list(chunk.squeeze(0).shape),
        all_actions_finite=bool(torch.isfinite(chunk).all()),
        first_chunk_seconds=round(first, 3),
        warm_chunk_seconds=round(warm, 3),
        action_min_per_joint=[round(v, 3) for v in chunk.amin(dim=(0, 1)).cpu().tolist()],
        action_max_per_joint=[round(v, 3) for v in chunk.amax(dim=(0, 1)).cpu().tolist()],
    )
    print(f'  chunk {tuple(chunk.shape)} in {warm:.2f}s (first {first:.2f}s) -> '
          f'{chunk.shape[1]} steps ≈ {chunk.shape[1] / args.fps:.2f}s of motion at {args.fps:g} fps', flush=True)


def make_robot(args):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    from lerobot.robots.so101_follower.so101_follower import SO101Follower

    cameras = {
        'fixed': OpenCVCameraConfig(
            index_or_path=args.camera_fixed, fps=args.camera_fps,
            width=args.width, height=args.height),
        'handeye': OpenCVCameraConfig(
            index_or_path=args.camera_handeye, fps=args.camera_fps,
            width=args.width, height=args.height),
    }
    config = SO101FollowerConfig(
        port=args.robot_port, id=args.robot_id, cameras=cameras,
        max_relative_target=args.max_relative_target,
        disable_torque_on_disconnect=True,
    )
    return SO101Follower(config)


def run_live(args, policy, config, result):
    import torch
    from lerobot.utils.robot_utils import busy_wait

    execute = args.execute
    robot = make_robot(args)
    print(f'connecting {args.robot_port} (id={args.robot_id}) and both cameras...', flush=True)
    frame_times, chunks, sent_steps = [], 0, 0
    clipped_steps = 0
    try:
        robot.connect()
        print('connected. Torque is ON; keep a hand near the power switch.', flush=True)
        with torch.inference_mode():
            policy.reset()
            start = time.perf_counter()
            step = 0
            while time.perf_counter() - start < args.seconds:
                loop_start = time.perf_counter()
                observation = robot.get_observation()
                batch = observation_to_batch(policy, config, observation, args.task, args.device)

                if args.snapshot_dir and step == 0:
                    save_snapshot(args.snapshot_dir, batch, config)

                predicts = step % config.n_action_steps == 0
                if predicts:
                    infer_start = time.perf_counter()
                # select_action generates a chunk when its queue is empty. Calling
                # predict_action_chunk here as well would generate and discard a
                # second chunk and double the work at each boundary.
                action = policy.select_action(batch).squeeze(0)
                if predicts:
                    if args.device == 'cuda':
                        torch.cuda.synchronize()
                    infer_seconds = time.perf_counter() - infer_start
                    chunks += 1

                if not torch.isfinite(action).all():
                    raise ValueError(f'policy produced non-finite action at step {step}: {action.tolist()}')

                goal = {f'{name}.pos': float(value) for name, value in
                        zip(('shoulder_pan', 'shoulder_lift', 'elbow_flex',
                             'wrist_flex', 'wrist_roll', 'gripper'), action.tolist(), strict=True)}

                if execute:
                    sent = robot.send_action(goal)
                    if any(abs(sent[k] - goal[k]) > 1e-4 for k in goal):
                        clipped_steps += 1
                    sent_steps += 1

                if predicts:
                    print(f'step {step:5d}  chunk {chunks} in {infer_seconds:5.2f}s  '
                          f'action=[{", ".join(f"{v:6.1f}" for v in action.tolist())}]'
                          f'{"  (dry run)" if not execute else ""}', flush=True)

                if step > 0:
                    frame_times.append(loop_start)
                step += 1
                busy_wait(1 / args.fps - (time.perf_counter() - loop_start))
    finally:
        # Never leave the arm energised on exit or Ctrl-C.
        try:
            robot.disconnect()
            print('robot disconnected.', flush=True)
        except Exception as error:
            print(f'disconnect problem: {type(error).__name__}: {error}', flush=True)

    intervals = [b - a for a, b in zip(frame_times, frame_times[1:])]
    result.update(
        mode='live-execute' if execute else 'live-dry-run',
        steps=step,
        chunks_predicted=chunks,
        actions_sent=sent_steps,
        actions_clipped_by_max_relative_target=clipped_steps,
        seconds=round(time.perf_counter() - start, 2),
        control_hz=round(1 / statistics.median(intervals), 2) if intervals else None,
        note='dry run: no action targets were sent; connect configures motors and enables torque' if not execute else 'arm was driven by the policy',
    )


def main() -> int:
    args = defaults()
    if args.fps <= 0 or args.seconds <= 0:
        raise ValueError('--fps and --seconds must be positive')
    if args.execute and args.mode != 'live':
        raise ValueError('--execute requires --mode live')
    if args.mode == 'live' and not args.robot_port:
        raise ValueError('--robot-port is required in live mode')
    if args.mode == 'live' and args.camera_fixed == args.camera_handeye:
        raise ValueError('fixed and handeye must use different camera indices')
    if args.list_cameras:
        return probe_cameras(args)
    result = {'ok': False, 'task': args.task}
    code = 0
    try:
        if args.mode == 'live' and not args.execute:
            print('LIVE DRY RUN: cameras and arm are read, but no action is sent.\n', flush=True)
        policy, config, info = load_policy(args)
        result.update(info)
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
        if args.mode == 'offline':
            run_offline(args, policy, config, result)
        else:
            run_live(args, policy, config, result)
        result['ok'] = True
    except KeyboardInterrupt:
        result['error'] = 'KeyboardInterrupt: stopped by user'
        code = 130
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc()
        code = 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'report: {args.output}', flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
