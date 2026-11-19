"""Offline SmolVLA compatibility check: no cameras, ports or robot connections."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]

def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=ROOT / 'outputs/smolvla/050000/pretrained_model')
    p.add_argument('--assets', type=Path, default=ROOT / 'assets/smolvlm')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--dataset', type=Path, help='Existing v2.1 dataset, opened read-only.')
    p.add_argument('--episode', type=int, default=1)
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--task', default='Pick up the green medicine box and place it in the black target area')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/check_result.json')
    return p.parse_args()

def video_rgb(path, timestamp, tolerance):
    import av
    best, delta = None, float('inf')
    with av.open(str(path), mode='r') as container:
        for frame in container.decode(video=0):
            if frame.time is None:
                continue
            distance = abs(frame.time - timestamp)
            if distance < delta:
                best, delta = frame.to_ndarray(format='rgb24'), distance
            if frame.time >= timestamp:
                break
    if best is None or delta > tolerance:
        raise ValueError(f'Unaligned video frame: {path}, difference={delta}')
    return best

def recorded_batch(args, config, torch):
    import pyarrow.parquet as pq
    root = args.dataset.resolve()
    info = json.loads((root / 'meta/info.json').read_text(encoding='utf-8'))
    if info['codebase_version'] != 'v2.1':
        raise ValueError('Read-only helper supports dataset v2.1 only')
    fields = dict(episode_chunk=args.episode // info['chunks_size'], episode_index=args.episode)
    table = pq.read_table(root / info['data_path'].format(**fields))
    if not 0 <= args.frame < table.num_rows:
        raise ValueError(f'Frame outside episode with {table.num_rows} rows')
    row = table.slice(args.frame, 1).to_pylist()[0]
    tasks = [json.loads(x) for x in (root / 'meta/tasks.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    task = next(x['task'] for x in tasks if x['task_index'] == row['task_index'])
    batch = {'observation.state': torch.tensor(row['observation.state'], dtype=torch.float32)[None], 'task': [task]}
    for key, feature in config.image_features.items():
        path = root / info['video_path'].format(**fields, video_key=key)
        rgb = video_rgb(path, float(row['timestamp']), 1 / info['fps'] + .001)
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255)[None]
        if list(image.shape[1:]) != list(feature.shape):
            raise ValueError(f'Invalid image shape {key}: {list(image.shape)}')
        batch[key] = image
    source = dict(kind='recorded', dataset=str(root), episode=args.episode, frame=args.frame, timestamp_s=float(row['timestamp']), task=task)
    return batch, source

def run(args, result):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    if args.repeats < 1:
        raise ValueError('--repeats must be positive')
    model, assets = args.model.resolve(), args.assets.resolve()
    for path in (model / 'config.json', model / 'model.safetensors', assets / 'config.json', assets / 'tokenizer.json', assets / 'processor_config.json', assets / 'preprocessor_config.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    names = ['torch', 'torchvision', 'transformers', 'tokenizers', 'huggingface-hub', 'accelerate', 'safetensors', 'lerobot']
    result.update(model=str(model), assets=str(assets), device=args.device, precision='float32', robot_connected=False, packages={x: metadata.version(x) for x in names})
    step = model.parent / 'training_state/training_step.json'
    if step.is_file():
        result['checkpoint_training_step'] = json.loads(step.read_text(encoding='utf-8'))['step']
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; --device cpu is an optional fallback.')
        gpu = torch.cuda.get_device_properties(0)
        result['gpu'] = dict(name=gpu.name, total_memory_mib=gpu.total_memory / 2**20, compute_capability=list(torch.cuda.get_device_capability()), torch_cuda_version=torch.version.cuda, native_bf16_supported=torch.cuda.is_bf16_supported(including_emulation=False))
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(1000)
    cfg = PreTrainedConfig.from_pretrained(model, local_files_only=True)
    if not isinstance(cfg, SmolVLAConfig):
        raise ValueError('Expected a SmolVLA checkpoint')
    cfg.device, cfg.vlm_model_name = 'cpu', str(assets)
    cfg.load_vlm_weights, cfg.use_amp = False, False
    started = time.perf_counter()
    print('Loading complete trained checkpoint on CPU, strict=True, FP32...', flush=True)
    policy = SmolVLAPolicy.from_pretrained(model, config=cfg, strict=True, local_files_only=True).float().eval()
    result['strict_weights_load'] = True
    normalizers = {}
    for key, tensor in policy.state_dict().items():
        if key.startswith(('normalize_inputs.', 'normalize_targets.', 'unnormalize_outputs.')):
            if not torch.isfinite(tensor).all() or (key.endswith('.std') and not (tensor > 0).all()):
                raise ValueError(f'Invalid normalization buffer: {key}')
            normalizers[key] = tensor.tolist()
    if len(normalizers) != 6:
        raise ValueError(f'Expected six normalization buffers; found {len(normalizers)}')
    result.update(normalization_valid=True, normalization_buffers=normalizers)
    if args.dataset:
        batch, result['input'] = recorded_batch(args, cfg, torch)
    else:
        mean = policy.state_dict()['normalize_inputs.buffer_observation_state.mean']
        batch = {'observation.state': mean[None].clone(), 'task': [args.task]}
        batch.update({k: torch.zeros((1, *v.shape), dtype=torch.float32) for k, v in cfg.image_features.items()})
        result['input'] = dict(kind='synthetic', task=args.task)
    if not torch.isfinite(batch['observation.state']).all():
        raise ValueError('Input state contains NaN or infinity')
    policy = policy.to(args.device)
    policy.config.device = args.device
    batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    result['load_seconds'] = time.perf_counter() - started
    result['parameter_dtypes'] = sorted({str(p.dtype) for p in policy.parameters()})
    if result['parameter_dtypes'] != ['torch.float32']:
        raise ValueError('Non-FP32 model parameters found')
    expected = (1, cfg.chunk_size, cfg.action_feature.shape[0])
    times = []
    with torch.inference_mode():
        for i in range(args.repeats):
            policy.reset()
            if args.device == 'cuda':
                torch.cuda.synchronize()
            started = time.perf_counter()
            actions = policy.predict_action_chunk(dict(batch))
            if args.device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            times.append(elapsed)
            if tuple(actions.shape) != expected or not torch.isfinite(actions).all():
                raise ValueError(f'Invalid action chunk: {tuple(actions.shape)}')
            print(f'Chunk {i+1}/{args.repeats}: {elapsed:.3f}s, shape={tuple(actions.shape)}, finite=True', flush=True)
        result.update(action_shape=list(actions.squeeze(0).shape), all_actions_finite=True, inference_seconds=times, warm_mean_chunk_seconds=sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0], action_min_per_joint=actions.amin(dim=(0, 1)).cpu().tolist(), action_max_per_joint=actions.amax(dim=(0, 1)).cpu().tolist())
        policy.reset()
        selected = [policy.select_action(dict(batch)) for _ in range(cfg.n_action_steps)]
        if any(tuple(a.shape) != (1, expected[2]) or not torch.isfinite(a).all() for a in selected):
            raise ValueError('Invalid select_action output')
        if len(policy._queues['action']) != 0:
            raise ValueError('Action queue did not drain')
        result.update(select_action_shape=[1, expected[2]], queued_actions_checked=len(selected))
    if args.device == 'cuda':
        result.update(cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20, cuda_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    digest = hashlib.sha256()
    with (model / 'model.safetensors').open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    result.update(checkpoint_sha256=digest.hexdigest(), ok=True, scope='Offline compatibility only; not physical grasp success or hardware safety.')

def main():
    args, result, code = arguments(), {'ok': False}, 0
    try:
        run(args, result)
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc()
        code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return code

if __name__ == '__main__':
    raise SystemExit(main())
