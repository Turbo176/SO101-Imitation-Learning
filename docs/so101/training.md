# Train and resume

The entry point is the retained `lerobot.scripts.train` implementation. `configs/so101/act_50k/` and `smolvla_50k/` archive the saved checkpoint configs with local paths made relative and a portable video backend. `configs/so101/train_*.json` are executable fresh-run recipes; they are not byte-for-byte copies of the historical configs.

ACT and SmolVLA were trained on a remote Linux GPU server. For the complete data-transfer, environment, persistent-session and checkpoint-return workflow, follow the [remote GPU training walkthrough](remote-training.md). The exact historical server GPU and full training environment lock are not available.

## Recorded settings

| Setting | ACT | SmolVLA |
| --- | --- | --- |
| Dataset | `local/medicine_box_black_5_20s_v1`, all 55 episodes | Same |
| Steps / batch size / seed | 50,000 / 4 / 1000 | Same |
| Observation history | Current observation | Current observation |
| Native chunk / executed chunk | 100 / 100 | 50 / 50 |
| Backbone | ResNet-18, ImageNet initialization setting | SmolVLM2-500M-Video-Instruct family |
| Optimizer | AdamW, lr `1e-5`, wd `1e-4` | AdamW, lr `1e-4`, wd `1e-10` |
| Schedule | None | 1,000-step warmup, 50,000-step cosine decay, final lr `2.5e-6` |
| Augmentation / AMP / WandB | Disabled / disabled / disabled | Same |
| Checkpoints | Every 10,000 steps | Every 5,000 steps |
| Saved eval configuration | `env=null`, `eval_freq=20000` | `env=null`, `eval_freq=0` |

ACT had no evaluation environment, so its nonzero evaluation frequency did not produce physical-success measurements. The SmolVLA 5k and 50k configs both plan 50,000 steps; the **actual saved step** comes from the neighboring `training_state/training_step.json`.

## Fresh training

```bash
python -m lerobot.scripts.train --config_path=configs/so101/train_act.json
python -m lerobot.scripts.train --config_path=configs/so101/train_smolvla.json
```

To use another dataset location:

```bash
python -m lerobot.scripts.train --config_path=configs/so101/train_act.json --dataset.root=../datasets/medicine_box_black_5_20s_v1 --output_dir=outputs/train/act_repeat
```

The fresh recipes set `resume=false`, `eval_freq=0` because there is no environment, and `video_backend=pyav` for Windows/Linux portability. The original training configs used `torchcodec`. ACT starts a new model with the saved architecture and ImageNet backbone initialization setting.

The fresh SmolVLA recipe explicitly sets `load_vlm_weights=true`, loading its pretrained visual-language backbone for a new run. The archived trained config has `load_vlm_weights=false`, which is appropriate when loading a complete checkpoint but **does not record how the original training run was initialized**. No base-model revision or initial checkpoint command survives. The fresh recipe is an explicit reconstruction choice; exact reproduction of the historical SmolVLA initialization is not claimed. Network access is required for pretrained backbone downloads unless the relevant models are cached or a local snapshot path is supplied as `--policy.vlm_model_name=...`.

Training requires the full local demonstrations. Weights and datasets are excluded from Git. New output directories must not already exist unless resuming. GPU memory requirements for a fresh run should be checked on the chosen host; historical training GPU identity was not saved in the retained records.

## Resume a saved run

Keep both sibling directories in the checkpoint:

```text
050000/
├── pretrained_model/{config.json,train_config.json,model.safetensors}
└── training_state/{training_step.json,optimizer_state.safetensors,...}
```

For ACT, using the current local checkpoint layout:

```bash
python -m lerobot.scripts.train --config_path=outputs/checkpoints/050000/pretrained_model/train_config.json --resume=true --dataset.root=../datasets/medicine_box_black_5_20s_v1 --dataset.video_backend=pyav --output_dir=outputs/train/act_resumed --steps=60000
```

For SmolVLA:

```bash
python -m lerobot.scripts.train --config_path=outputs/smolvla/050000/pretrained_model/train_config.json --resume=true --dataset.root=../datasets/medicine_box_black_5_20s_v1 --dataset.video_backend=pyav --output_dir=outputs/train/smolvla_resumed --policy.vlm_model_name=assets/smolvlm --policy.load_vlm_weights=false --steps=60000
```

Prepare the SmolVLM assets using the [deployment guide](deployment.md) first. The trainer finds the weights and optimizer state relative to the actual `train_config.json` passed in `--config_path`; an archived config under `configs/` cannot replace this checkpoint layout. For an exact interrupted-run continuation, retain the original planned step count and scheduler; extending to 60k is a new training decision and may need a deliberate schedule change.

The release checks parse these entry points/configurations and load existing weights without commanding hardware. A new 50k training run and optimizer-resume updates are not performed as part of publication preparation.
