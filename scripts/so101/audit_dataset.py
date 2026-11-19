"""Read a LeRobot v2.1 dataset, validate aligned records, and optionally decode every video."""
import argparse
import hashlib
import json
from pathlib import Path

def audit(root, decode_videos=False):
    import numpy as np
    import pyarrow.parquet as pq
    info = json.loads((root / 'meta/info.json').read_text(encoding='utf-8'))
    episodes = [json.loads(x) for x in (root / 'meta/episodes.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    if info['codebase_version'] != 'v2.1':
        raise ValueError('This project uses LeRobot v2.1, not v3.')
    cameras = [k for k, v in info['features'].items() if v['dtype'] == 'video']
    rows, failures, video_count = [], [], 0
    for episode in episodes:
        i = episode['episode_index']
        fields = dict(episode_index=i, episode_chunk=i // info['chunks_size'])
        path = root / info['data_path'].format(**fields)
        table = pq.read_table(path)
        frame_index = np.array(table['frame_index'].to_pylist())
        times = np.array(table['timestamp'].to_pylist())
        state = np.asarray(table['observation.state'].to_pylist())
        action = np.asarray(table['action'].to_pylist())
        checks = {'length_matches': table.num_rows == episode['length'],
                  'frame_index_contiguous': bool(np.array_equal(frame_index, np.arange(table.num_rows))),
                  'timestamps_match_fps': bool(np.allclose(times, frame_index / info['fps'], atol=1e-5)),
                  'six_joint_shape': state.shape == action.shape == (table.num_rows, 6),
                  'all_finite': bool(np.isfinite(state).all() and np.isfinite(action).all())}
        row = dict(episode_index=i, frames=table.num_rows, checks=checks, videos={})
        for key in cameras:
            video = root / info['video_path'].format(**fields, video_key=key)
            count = None
            if not video.exists():
                failures.append(f'episode {i}: missing {key}')
            else:
                video_count += 1
                if decode_videos:
                    import av
                    try:
                        with av.open(str(video)) as container:
                            stream = container.streams.video[0]
                            count = sum(1 for _ in container.decode(video=0))
                            if count != table.num_rows or (stream.height, stream.width, 3) != tuple(info['features'][key]['shape']):
                                failures.append(f'episode {i}: video shape/count mismatch for {key}')
                    except Exception as error:
                        failures.append(f'episode {i}: {key}: {error}')
            row['videos'][key] = {'exists': video.exists(), 'decoded_frames': count}
        failures.extend(f'episode {i}: {k}' for k, v in checks.items() if not v)
        rows.append(row)
    frames = sum(x['frames'] for x in rows)
    if len(rows) != info['total_episodes'] or frames != info['total_frames']:
        failures.append('dataset totals differ from metadata')
    return {'ok': not failures, 'format': info['codebase_version'], 'episodes': len(rows),
            'frames': frames, 'fps': info['fps'], 'videos': video_count,
            'decoded_videos': decode_videos, 'splits': info['splits'],
            'info_sha256': hashlib.sha256((root / 'meta/info.json').read_bytes()).hexdigest(),
            'failures': failures, 'episode_checks': rows}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--decode-videos', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('outputs/dataset_audit.json'))
    args = parser.parse_args()
    result = audit(args.root, args.decode_videos)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'episode_checks'}, indent=2))
    return 0 if result['ok'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
