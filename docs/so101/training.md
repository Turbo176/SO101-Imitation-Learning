# Training ACT and SmolVLA

Both policies were trained on a remote Linux GPU server using the same 55 demonstrations. The robot computer handles collection and deployment; the server needs code and data, not robot calibration or a serial connection.

## Settings

| Setting | ACT | SmolVLA |
| --- | --- | --- |
| Steps / batch size / seed | 50,000 / 4 / 1000 | Same |
| Observation | Current two-view images and joint state | Same, plus task text |
| Action chunk | 100 | 50 |
| Backbone | ResNet-18, ImageNet initialization | SmolVLM2-500M-Video-Instruct family |
| AdamW learning rate / weight decay | `1e-5` / `1e-4` | `1e-4` / `1e-10` |
| Schedule | None | 1,000-step warmup, cosine decay to `2.5e-6` |
| Checkpoint interval | 10,000 | 5,000 |
| Augmentation / AMP / WandB | Disabled | Disabled |

`configs/so101/*_50k/` contains the saved checkpoint configurations with portable paths. `train_act.json` and `train_smolvla.json` start fresh runs using PyAV, `resume=false` and no evaluation environment. The actual completed step is stored in `training_state/training_step.json`.

The original GPU, full Linux environment lock and SmolVLA initialization command were not saved. The workflow below specifies a compatible new run. The retained `gradient_accumulation_steps` field is config metadata; this trainer has no accumulation loop and the recorded value is 1.

<a id="remote-training"></a>
## Transfer code and data

On the server:

```bash
mkdir -p ~/so101-work
cd ~/so101-work
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Turbo176/SO101-Imitation-Learning.git
cd SO101-Imitation-Learning
git rev-parse HEAD
mkdir -p ../datasets
```

Reuse an existing checkout and dataset when available. From the repository root on the robot computer, upload the complete dataset using your own SSH alias, here named `training-server`:

```bash
tar -cf medicine_box_black_5_20s_v1.tar -C ../datasets medicine_box_black_5_20s_v1
scp medicine_box_black_5_20s_v1.tar training-server:~/so101-work/
```

From the repository root on the server:

```bash
tar -xf ../medicine_box_black_5_20s_v1.tar -C ../datasets
```

Keep `meta/`, `data/` and `videos/` together. SSH transfer needs no HTTP proxy unless your own SSH configuration uses one. A provider's resumable transfer tool also works.

## Server environment and pretrained assets

Activate a Python 3.10 environment. For a new environment, create it with `python3.10 -m venv .venv` and `source .venv/bin/activate`. Then install a PyTorch wheel supported by the server driver:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[smolvla]" -c requirements.txt
python -m pip check
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python scripts/so101/audit_dataset.py --root ../datasets/medicine_box_black_5_20s_v1 --decode-videos
```

ACT downloads ResNet-18 initialization weights on first use unless cached. A fresh SmolVLA run needs the full pretrained backbone:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolVLM2-500M-Video-Instruct', revision='7b375e1b73b11138ff12fe22c8f2822d8fe03467', local_dir='assets/smolvlm-training')"
```

This revision fixes the backbone for the new recipe. It does not identify the historical initialization. `prepare_assets.py` downloads only processor/config files for completed checkpoints; it cannot replace the backbone weights needed for fresh training. Reuse a complete cache when available.

## Start training

Run one job at a time on a single GPU. Choose a new output directory for every run.

```bash
python -u -m lerobot.scripts.train \
  --config_path=configs/so101/train_act.json \
  --dataset.root=../datasets/medicine_box_black_5_20s_v1 \
  --policy.device=cuda --output_dir=outputs/train/act_cloud_50k
```

```bash
python -u -m lerobot.scripts.train \
  --config_path=configs/so101/train_smolvla.json \
  --dataset.root=../datasets/medicine_box_black_5_20s_v1 \
  --policy.device=cuda --policy.vlm_model_name=assets/smolvlm-training \
  --policy.load_vlm_weights=true --output_dir=outputs/train/smolvla_cloud_50k
```

Keep `fixed` and `handeye` mapped to the same views. Reduce batch size if memory requires it and record that change with the new run. All 55 episodes were used in the original training; a held-out split requires a new training run.

For an SSH session, start `tmux new -s so101-act`, activate the environment inside it, and log the training command with `2>&1 | tee logs/act_cloud_50k.log`. Create `logs/` first and use `set -o pipefail`. Detach with `Ctrl-b`, `d`; reconnect with `tmux attach -t so101-act`.

```bash
tail -f logs/act_cloud_50k.log
python -m pip freeze > logs/act_cloud_50k.environment.txt
git rev-parse HEAD > logs/act_cloud_50k.commit.txt
```

The trainer writes terminal progress and checkpoints. The result CSVs in this repository are separate experiment exports.

## Resume

Keep the whole numbered checkpoint, including both directories:

```text
030000/
├── pretrained_model/{config.json,train_config.json,model.safetensors}
└── training_state/{training_step.json,optimizer_state.safetensors,...}
```

For an interrupted ACT run:

```bash
python -u -m lerobot.scripts.train \
  --config_path=outputs/train/act_cloud_50k/checkpoints/030000/pretrained_model/train_config.json \
  --resume=true --dataset.root=../datasets/medicine_box_black_5_20s_v1 \
  --dataset.video_backend=pyav --output_dir=outputs/train/act_cloud_50k
```

For SmolVLA:

```bash
python -u -m lerobot.scripts.train \
  --config_path=outputs/train/smolvla_cloud_50k/checkpoints/025000/pretrained_model/train_config.json \
  --resume=true --dataset.root=../datasets/medicine_box_black_5_20s_v1 \
  --dataset.video_backend=pyav --policy.vlm_model_name=assets/smolvlm-training \
  --policy.load_vlm_weights=false --output_dir=outputs/train/smolvla_cloud_50k
```

Use a checkpoint that exists. The trainer locates weights and optimizer state relative to that checkpoint's `train_config.json`; the config copies under `configs/` are insufficient. Keep the original step target and scheduler for an interrupted-run continuation.

## Bring a checkpoint back

On the server, package a complete checkpoint and record its hash:

```bash
tar -cf act_050000.tar -C outputs/train/act_cloud_50k/checkpoints 050000
sha256sum act_050000.tar
```

On the robot computer:

```bash
scp training-server:~/so101-work/SO101-Imitation-Learning/act_050000.tar .
mkdir -p outputs/imported/act_cloud_50k
tar -xf act_050000.tar -C outputs/imported/act_cloud_50k
```

For SmolVLA, use `smolvla_cloud_50k` and a separate archive/import directory. Compare the transferred hash using `sha256sum` or PowerShell `Get-FileHash`. In PowerShell, create folders with `New-Item -ItemType Directory -Force` instead of `mkdir -p`.

Pass the imported `050000/pretrained_model` path to the [deployment tools](deployment.md). A complete checkpoint includes normalization buffers and training state; physical calibration and SmolVLA processor assets remain separate. Keep full datasets, checkpoints and server settings outside Git.
