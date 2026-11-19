"""Plot retained loss exports and matched training-observation diagnostics."""
import argparse
import csv
from pathlib import Path
import numpy as np

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/so101'))
    parser.add_argument('--output', type=Path, default=Path('media/so101/training-results.svg'))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'path', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': '#334c64',
                         'text.color': '#1b2c43', 'xtick.color': '#526d85', 'ytick.color': '#526d85'})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), layout='constrained')
    for ax, label, filename, color in zip(axes[:2], ['ACT', 'SmolVLA'],
             ['act_training_metrics.csv', 'smolvla_metrics_200step.csv'], ['#cc6461', '#397fb9']):
        rows = read(args.results / filename)
        x = np.array([float(r['step']) for r in rows]); y = np.array([float(r['loss']) for r in rows])
        ax.plot(x, y, color=color, alpha=.22, lw=.9, label='200-step window')
        ax.plot(x[4:], np.convolve(y, np.ones(5) / 5, mode='valid'), color=color, lw=2,
                label='1,000-step rolling mean')
        ax.set_yscale('log'); ax.set_title(f'{label} training loss', fontweight='bold')
        ax.set_xlabel('Training step');ax.set_ylabel('Loss (log scale)')
        ax.grid(alpha=.16);ax.legend(frameon=False, fontsize=8)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(3,3))
    rows = read(args.results / 'checkpoint_metrics.csv')
    for label, color in [('ACT', '#cc6461'), ('SmolVLA', '#397fb9')]:
        selected = [r for r in rows if r['model'] == label]
        axes[2].plot([float(r['step']) for r in selected], [float(r['normalized_mae']) for r in selected],
                     color=color, marker='o', lw=2, label=label)
    axes[2].set_title('Matched training observations', fontweight='bold')
    axes[2].set_xlabel('Checkpoint step');axes[2].set_ylabel('Normalized MAE')
    axes[2].ticklabel_format(axis='x', style='sci', scilimits=(3,3));axes[2].grid(alpha=.16)
    axes[2].legend(frameon=False)
    fig.suptitle('Separate loss objectives · 15 seen observations · no held-out or robot-success estimate',fontsize=10,color='#526d85')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight', metadata={'Date': None})
    # Matplotlib inserts trailing spaces in SVG path coordinates.
    args.output.write_text('\n'.join(line.rstrip() for line in args.output.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    fig.savefig(args.output.with_suffix('.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)
    # Show the preselected middle observation, rather than choosing a best fit.
    arrays = np.load(args.results / 'checkpoint_predictions.npz', allow_pickle=False)
    observation = len(arrays['target']) // 2
    metadata = next(row for row in read(args.results / 'checkpoint_observation_metrics.csv')
                    if row['model'] == 'ACT' and int(row['step']) == 50000
                    and int(row['observation_id']) == observation + 1)
    checkpoint = int(np.flatnonzero(arrays['steps'] == 50000)[0])
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 5.7), layout='constrained')
    for joint, ax in enumerate(axes.flat):
        ax.plot(arrays['time'], arrays['target'][observation, :, joint],
                color='#18374b', lw=2.4, label='Recorded action')
        for name, color in [('ACT', '#cc6461'), ('SmolVLA', '#397fb9')]:
            ax.plot(arrays['time'], arrays[name.lower()][checkpoint, observation, :, joint],
                    color=color, lw=1.7, label=f'{name} 50k')
        ax.set_title(str(arrays['joint_names'][joint]).replace('_', ' '), fontweight='bold')
        ax.set_xlabel('Future time (s)')
        ax.set_ylabel('Recorded position units')
        ax.grid(alpha=.16)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle(f'Seen training observation {observation + 1} / episode {metadata["episode"]} '
                 f'/ fraction {metadata["episode_fraction"]}\n'
                 'One 50-step prediction chunk per model; no executed robot trajectory',
                 fontsize=12, color='#18374b')
    action_output = args.output.with_name('action-chunks.svg')
    fig.savefig(action_output, bbox_inches='tight', metadata={'Date': None})
    action_output.write_text('\n'.join(line.rstrip() for line in action_output.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    fig.savefig(action_output.with_suffix('.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(args.output)
    print(action_output)

if __name__ == '__main__':
    main()
