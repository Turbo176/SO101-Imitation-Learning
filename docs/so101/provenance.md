# Provenance and project history

## Existing repository

This source release comes from the existing LeRobot checkout at upstream base `882c80d446a63a44868c67ae535467af32ce0e80`; its package declares LeRobot 0.3.4. It includes the existing SO-101 experiment changes. Upstream copyright headers, Apache-2.0 licensing and attribution are retained. The original top-level README is preserved at [docs/upstream/README.md](../upstream/README.md), and the base remains identifiable in the upstream LeRobot repository.

The supplied Git history contained upstream commits but no committed personal experiment changes. The public project is distributed as an independent source snapshot rather than importing the upstream commit graph into its contributor list. The complete original history and the publication-preparation commits are retained in a local archive. Snapshot publication does not transfer authorship of the upstream code: the notices and third-party attribution below remain authoritative. A new commit date cannot establish an old experiment date; the surviving source dates and logs below establish the available timeline.

The current upstream test fixtures and selected publication media are available from this repository. Large test fixtures retain their upstream LFS handling. Full experiment datasets and trained policy weights are excluded. The public snapshot does not include the upstream historical commit graph; consult the upstream repository for that development history.

## Recorded timeline

| Date | Evidence | What it establishes |
| --- | --- | --- |
| September 2, 2025 | Original clone reflog and upstream base | Checkout lineage; not the experiment completion date |
| July–September 2026 | Preserved modification times on SO-101 integration, camera, and serial changes | Local source snapshots existed before the training/analysis period; exact original commit dates were not recorded |
| September 29, 2026 | ACT training CSV timestamps, resumed training config, neighboring 50k training state | ACT trained through 50,000 steps; terminal timestamps have no explicit timezone |
| September 29, 2026 | Deployment source videos' preserved local modification times | Existing deployment recordings, including visible pick-and-place; not independently authenticated capture timestamps |
| October 7, 2026 | SmolVLA training CSV with explicit `+00:00` timestamps | SmolVLA training progressed through 50,000 steps |
| October 7–10, 2026 | Existing offline-check JSONs and inference scripts | Windows FP32 compatibility checks and local deployment tooling |
| October 11–13, 2026 | Saved predictions, metric CSVs, analysis reports and their preserved times | Post-training diagnostics and data/figure audit materials |
| November 19, 2026 | Publication preparation | English documentation, portable defaults, selected-media exports, source manifest, and current verification |

The project is described as a September–October 2026 experiment. The November publication cleanup is recorded separately. File modification times support the source snapshot chronology; they are not converted into invented original Git commits. Archived import commits distinguish supported source author dates from their November import dates. The public source snapshot and newly authored presentation work use their actual November publication date.

### Dates shown on GitHub

This repository was first published on November 19, 2026. README updates and the synchronized telemetry presentation were also authored on that date. GitHub's repository view uses commit dates, while the profile contribution timeline uses author dates; these can differ for imported work. The file list reflects the relevant Git commits, not old filesystem modification times or the dates of the underlying experiment. A file whose documentation changed on November 19 can therefore show that date even when it describes September–October work. The timeline above and retained source records establish the experiment chronology. The public branch was converted to an independent source snapshot on the same publication date; this changes history packaging without making the release appear older.

GitHub documents the distinction in [“Why are my contributions not showing up on my profile?”](https://docs.github.com/en/account-and-profile/how-tos/setting-up-and-managing-your-github-profile/managing-contribution-settings/troubleshooting-missing-contributions), under “How GitHub uses the Git author date and commit date.”

## Retained local changes

| Area | Existing implementation retained |
| --- | --- |
| OpenCV camera | Optional capture FOURCC, requested-mode handling, related test additions |
| SO-101 / motor buses | Serial position-read and startup-configuration retries; follower/leader propagation |
| Recording | Empty-buffer early-exit handling |
| Policy config | Windows-compatible temporary config loading; existing config compatibility field |
| Robot factories / CLI registration | Existing dual-SO101 integration and registrations |
| SmolVLA deployment | `run_policy.py`, `check_model.py`, pinned compatibility requirements |
| Analysis | Existing loss exports, continuous-action metric code, saved prediction arrays |

The local `gradient_accumulation_steps` config field is retained for checkpoint compatibility. The current trainer does not implement an accumulation loop for that field; all archived runs used 1. Dual-arm integrations are present but are not a result of the single-arm box experiment.

Publication adaptations move the ignored inference scripts into `scripts/so101/`, use repository-relative defaults, require an explicit live serial port, map raw camera names to policy keys, correct a redundant chunk-generation call, and clarify that dry mode still performs startup motor configuration. New audits and plots operate on retained data and result artifacts. The revised live runner is verified without hardware; no new physical evaluation is claimed.

The source checkpoint/config hashes are in `results/so101/checkpoint_manifest.json`. `source_artifacts.json` records hashes, source-relative names, and preserved dates of imported experiment materials. Config archives replace machine paths and select PyAV for portability; fresh recipes also reset resume/evaluation fields and explicitly initialize SmolVLA's backbone. Original config hashes remain available so a portable copy is not confused with the source.

## Third-party material

| Material | Attribution / handling |
| --- | --- |
| LeRobot 0.3.4 source | Hugging Face and contributors; Apache-2.0; LICENSE and original headers retained |
| ACT implementation and algorithm | Upstream LeRobot implementation; cite ALOHA / ACT authors |
| SmolVLA implementation and algorithm | Upstream LeRobot implementation; cite SmolVLA authors |
| SmolVLM2 processor/config files | Model card declares Apache-2.0; download from the named upstream model at the fixed revision; excluded from Git |
| ResNet-18 pretrained weights | Downloaded dependency used by the recorded ACT setting; excluded from Git |
| Dataset and experiment videos | Supplied physical experiment recordings; selected footage is labeled and hashed; full dataset excluded |
| README assets | New local SVG diagrams/badges and plots generated from this experiment's existing CSVs |

The reference repositories Pi05-Mujoco and Pi05-FR3-Gello were read for page organization and presentation only. Their source, assets, datasets, and performance results are not incorporated.

No secret credentials, server connection commands, environments, full datasets, or training weights belong in the publication content. Unrelated upstream publishing workflows are archived as inert examples for this release; project checks do not publish the package under the upstream PyPI name.
