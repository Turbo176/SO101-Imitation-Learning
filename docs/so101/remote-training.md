# Train ACT and SmolVLA on a remote GPU server

Both policies in this experiment were trained on a remote Linux GPU server. The robot computer collected demonstrations and later loaded trained checkpoints for physical deployment. This guide uses the existing trainer and exported configurations in this repository.

The retained records establish the dataset, policy settings, training steps and checkpoint state. They do not establish the original server GPU, a complete Linux dependency lock, or the original SmolVLA initialization command. The commands below provide a compatible fresh-run workflow; they do not claim a byte-for-byte reconstruction of the historical environment.

## 1. Keep the robot and training roles separate

| Robot computer | Linux GPU server |
| --- | --- |
| Leader/follower, cameras and physical calibration | Compatible NVIDIA driver, CUDA-enabled PyTorch and local dataset copy |
| Demonstration collection and dataset audit | ACT or SmolVLA training and checkpoint storage |
| Offline checkpoint check, then physical deployment | Logs, optimizer/scheduler state and exported configuration |

Use the same repository revision and the same recorded camera names, joint order and dataset statistics. Training does not need a robot serial connection. Keep private connection settings and physical calibration outside Git.

Examples use an SSH alias named `training-server`, configured on your own computer. Substitute your existing alias or use your provider's file-transfer interface. No server address or login credentials are supplied by this repository.

## 2. Put code and data on the server

In an existing server terminal:

```bash
mkdir -p ~/so101-work
cd ~/so101-work
# Training does not need the upstream LFS test fixtures.
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Turbo176/SO101-Imitation-Learning.git
cd SO101-Imitation-Learning
git rev-parse HEAD
mkdir -p ../datasets
```

If this checkout already exists on the server, reuse it and verify its revision instead of cloning or reinstalling it. Preserve local changes before updating code.

On the robot computer, from this repository root, package the existing dataset without altering its layout:

```bash
tar -cf medicine_box_black_5_20s_v1.tar -C ../datasets medicine_box_black_5_20s_v1
scp medicine_box_black_5_20s_v1.tar training-server:~/so101-work/
```

On the server, from `~/so101-work/SO101-Imitation-Learning`:

```bash
tar -xf ../medicine_box_black_5_20s_v1.tar -C ../datasets
```

Activate the environment in the next section, then run the dataset audit there. The expected dataset has **55 episodes, 20,512 frames and 110 videos**, with `meta/`, `data/` and `videos/` together. Uploading only the MP4 files loses task labels, joint targets and normalization statistics. For a large upload, your provider's resumable file-transfer facility can be used instead of `scp`.

The `scp` commands use your SSH connection. They do not require an HTTP proxy. Reuse a dataset already on the server rather than transferring another copy. Model downloads use HTTPS; use an existing complete cache when available.

```text
so101-work/
├── SO101-Imitation-Learning/
│   ├── configs/so101/
│   ├── src/lerobot/
│   └── outputs/train/             Generated on the server; ignored by Git
└── datasets/
    └── medicine_box_black_5_20s_v1/
        ├── meta/
        ├── data/
        └── videos/
```

## 3. Reuse or prepare a compatible environment

Activate a working Python 3.10 environment when one is already available. For a new environment only:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Select a wheel compatible with the server's NVIDIA driver.
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[smolvla]" -c requirements-so101.txt
python -m pip check
```

The CUDA 12.8 wheels above are the compatibility profile checked with the retained deployment environment. Verify driver support on your chosen server; that wheel choice is not evidence of the original Linux training environment. The server does not need the optional Feetech serial package for training alone.

```bash
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python scripts/so101/audit_dataset.py --root ../datasets/medicine_box_black_5_20s_v1 --decode-videos
```

Start with the recorded batch size of 4 and FP32 setting. If they exceed the selected GPU's memory, reduce batch size deliberately and record the changed setting. A different batch size is a new experiment. The retained `gradient_accumulation_steps` field is compatibility metadata; this trainer does not implement an accumulation loop for it.

## 4. Prepare pretrained assets once

ACT's fresh recipe uses the ImageNet ResNet-18 initialization setting. Torchvision downloads those backbone weights on first use unless they are already cached. It does not download a completed policy for this task.

SmolVLA's fresh recipe loads the pretrained visual-language backbone. To keep that download separate from training and record a fixed revision:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolVLM2-500M-Video-Instruct', revision='7b375e1b73b11138ff12fe22c8f2822d8fe03467', local_dir='assets/smolvlm-training')"
```

This is a **full backbone snapshot including weights**. The smaller `scripts/so101/prepare_assets.py` command downloads only processor/config files for loading a completed checkpoint offline; those small files cannot initialize a fresh SmolVLA training run.

`assets/smolvlm-training/` is a local asset directory and must stay outside Git. The revision above is an explicit choice for this reproducible recipe. The historical SmolVLA initialization source was not saved. Reuse an existing complete compatible snapshot when available and record its identity.

## 5. Train one policy at a time

Run commands from the repository root. Select a new output directory for each fresh run. The configurations retain batch size 4, seed 1000, 50,000 steps, two camera inputs, six joints and disabled augmentation/AMP/WandB. The portable recipes use PyAV instead of the original saved TorchCodec backend.

ACT:

```bash
python -u -m lerobot.scripts.train \
  --config_path=configs/so101/train_act.json \
  --dataset.root=../datasets/medicine_box_black_5_20s_v1 \
  --policy.device=cuda \
  --output_dir=outputs/train/act_cloud_50k
```

SmolVLA, with the complete local backbone snapshot from the previous step:

```bash
python -u -m lerobot.scripts.train \
  --config_path=configs/so101/train_smolvla.json \
  --dataset.root=../datasets/medicine_box_black_5_20s_v1 \
  --policy.device=cuda \
  --policy.vlm_model_name=assets/smolvlm-training \
  --policy.load_vlm_weights=true \
  --output_dir=outputs/train/smolvla_cloud_50k
```

Keep the recorded camera assignment: `fixed` is the tabletop view and `handeye` is the gripper view. Keep all 55 episodes for this historical training recipe. A new held-out split should be specified and reported as a new evaluation design; the existing diagnostics use seen training observations.

### Keep training alive after an SSH disconnect

An existing provider terminal manager is sufficient. With `tmux` installed:

```bash
tmux new -s so101-act
```

Inside that session, reactivate the environment if necessary, enter the repository, then run:

```bash
mkdir -p logs
set -o pipefail
python -u -m lerobot.scripts.train \
  --config_path=configs/so101/train_act.json \
  --output_dir=outputs/train/act_cloud_50k 2>&1 | tee logs/act_cloud_50k.log
```

Detach with `Ctrl-b`, then `d`; reconnect with `tmux attach -t so101-act`. Use a separate session and log filename for SmolVLA, substituting its complete command above. On a single GPU, finish or stop the current job before starting the other model.

This checkout writes training progress to the terminal and checkpoints to the output directory. The original metrics CSVs in `results/so101/` are retained experiment exports; a new training run does not automatically overwrite them. Keep its own log, config, repository revision and environment record.

```bash
tail -f logs/act_cloud_50k.log
python -m pip freeze > logs/act_cloud_50k.environment.txt
git rev-parse HEAD > logs/act_cloud_50k.commit.txt
```

## 6. Resume an interrupted run

Each checkpoint contains both model files and training state. ACT saves every 10,000 steps; SmolVLA every 5,000. The final step is 50,000 for both.

```text
outputs/train/act_cloud_50k/checkpoints/030000/
├── pretrained_model/
│   ├── config.json
│   ├── train_config.json
│   └── model.safetensors
└── training_state/
    ├── training_step.json
    ├── optimizer_state.safetensors
    └── ...                         RNG / optimizer groups / scheduler state
```

For an ACT run interrupted after the 30,000-step save:

```bash
python -u -m lerobot.scripts.train \
  --config_path=outputs/train/act_cloud_50k/checkpoints/030000/pretrained_model/train_config.json \
  --resume=true \
  --dataset.root=../datasets/medicine_box_black_5_20s_v1 \
  --dataset.video_backend=pyav \
  --output_dir=outputs/train/act_cloud_50k
```

For SmolVLA interrupted after the 25,000-step save:

```bash
python -u -m lerobot.scripts.train \
  --config_path=outputs/train/smolvla_cloud_50k/checkpoints/025000/pretrained_model/train_config.json \
  --resume=true \
  --dataset.root=../datasets/medicine_box_black_5_20s_v1 \
  --dataset.video_backend=pyav \
  --policy.vlm_model_name=assets/smolvlm-training \
  --policy.load_vlm_weights=false \
  --output_dir=outputs/train/smolvla_cloud_50k
```

Use a checkpoint that actually exists. Passing its own `train_config.json` lets this trainer locate the sibling weights and optimizer state. The config archives under `configs/so101/` alone cannot resume a run. Keep the original planned 50,000 steps and scheduler for interrupted-run continuation; increasing the target step count changes the experiment.

## 7. Return a complete checkpoint to the robot computer

On the server, package the whole numbered checkpoint. For ACT:

```bash
tar -cf act_050000.tar -C outputs/train/act_cloud_50k/checkpoints 050000
sha256sum act_050000.tar
```

For SmolVLA:

```bash
tar -cf smolvla_050000.tar -C outputs/train/smolvla_cloud_50k/checkpoints 050000
sha256sum smolvla_050000.tar
```

On the robot computer, from this repository root:

```bash
scp training-server:~/so101-work/SO101-Imitation-Learning/act_050000.tar .
scp training-server:~/so101-work/SO101-Imitation-Learning/smolvla_050000.tar .
mkdir -p outputs/imported/act_cloud_50k outputs/imported/smolvla_cloud_50k
tar -xf act_050000.tar -C outputs/imported/act_cloud_50k
tar -xf smolvla_050000.tar -C outputs/imported/smolvla_cloud_50k
```

Use a distinct import directory for every new run. The paths above preserve any existing experiment checkpoints. The `mkdir -p` line uses Bash; in PowerShell create the same folders with `New-Item -ItemType Directory -Force outputs/imported/act_cloud_50k, outputs/imported/smolvla_cloud_50k`. Check the transferred archive's SHA256 using `sha256sum` or PowerShell `Get-FileHash` before extracting.

After preparing the small processor assets described in the deployment guide, check the imported SmolVLA checkpoint without commanding hardware:

```bash
python scripts/so101/check_model.py \
  --model outputs/imported/smolvla_cloud_50k/050000/pretrained_model \
  --assets assets/smolvlm \
  --dataset ../datasets/medicine_box_black_5_20s_v1 \
  --episode 25 --frame 60 --device cuda \
  --output outputs/imported/smolvla_cloud_50k/check.json
```

Use `outputs/imported/act_cloud_50k/050000/pretrained_model` as `--policy.path` for ACT deployment, and `outputs/imported/smolvla_cloud_50k/050000/pretrained_model` as `--model` for the SmolVLA runner. Apply the camera, port and calibration settings in the deployment guide before enabling physical motion.

The full checkpoint retains the saved normalization buffers and model config, plus training state for later resume. It does not contain physical robot calibration. SmolVLA's tokenizer/processor assets must also be available locally; follow the [offline check and deployment guide](deployment.md) before enabling physical motion. Use the robot's matching calibration, camera mapping and joint order.

Archive logs, complete checkpoints and the run environment separately from the public source repository. The project excludes full demonstrations, trained weights, private calibration and personal server settings from Git.
