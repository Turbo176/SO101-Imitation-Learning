"""Recompute the published diagnostic CSV from the original compact prediction arrays."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from metrics import action_metrics, curve_row

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/so101'))
    args = parser.parse_args()
    arrays = np.load(args.results / 'checkpoint_predictions.npz', allow_pickle=False)
    expected = list(csv.DictReader((args.results / 'checkpoint_metrics.csv').open(encoding='utf-8-sig')))
    worst, rows = 0., 0
    for name in ['ACT', 'SmolVLA']:
        for i, step in enumerate(arrays['steps']):
            calculated = curve_row(name, int(step), action_metrics(arrays[name.lower()][i], arrays['target'], arrays['action_std']))
            saved = next(x for x in expected if x['model'] == name and int(x['step']) == step)
            for key, value in calculated.items():
                if key != 'model':
                    difference = abs(float(saved[key]) - value)
                    worst = max(worst, difference)
                    if difference > 1e-10:
                        raise AssertionError(f'{name}/{step}/{key}: {saved[key]} != {value}')
            rows += 1
    if rows != len(expected):
        raise AssertionError('Unexpected extra rows in checkpoint_metrics.csv')
    print(json.dumps({'ok': True, 'checkpoints': rows, 'predicted_chunks': 150,
                      'maximum_absolute_difference': worst, 'scope': 'training observations'}, indent=2))

if __name__ == '__main__':
    main()
