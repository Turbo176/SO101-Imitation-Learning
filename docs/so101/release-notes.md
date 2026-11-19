# Source release notes

This release packages the existing September–October 2026 SO-101 experiment with English documentation and local presentation assets.

- Publishes an independent snapshot of the existing LeRobot 0.3.4 checkout, retaining the upstream base reference, license notices and source attribution. The complete preparation history remains archived locally.
- Preserves local camera, SO-101 retry, recording, and policy-config changes.
- Migrates the existing ignored SmolVLA inference scripts into versioned source and makes their defaults portable.
- Removes a redundant chunk-prediction call from the live runner; `select_action` already refills its queue.
- Maps raw follower camera names to policy observation keys and attempts disconnect cleanup if startup fails.
- Includes source hashes, dataset audits, loss exports, saved diagnostic arrays, and result-recomputation tools.
- Publishes selected recordings with control/source labels, no time cuts, and native plus clearly labeled display copies.
- Shows fixed and handeye recordings together, with an explicit card when the trial06 handeye stream ends; adds full-interval demonstration and failure previews, real task keyframes, and a six-joint diagnostic plot.
- Adds synchronized real-camera playback with the same trial's measured joint positions and sent targets, a selected 563-row sample table, and checked timestamp alignment.
- Centers every row of the homepage section navigation and folds secondary camera recordings.
- Archives upstream deployment/PyPI workflows as inert examples and adds project-specific source/result checks.

The source distribution does not include environments, the full dataset, model weights, private calibration, or server connection details. The Apache-2.0 upstream license and attribution are retained. The package identity remains `lerobot` for compatibility; this project does not publish a replacement package under that name.

Historical physical success counts and exact per-trial controller identities remain unrecorded. The release does not retroactively turn selected footage or training diagnostics into an autonomous benchmark. No new robot trial or full retraining was run for publication.
