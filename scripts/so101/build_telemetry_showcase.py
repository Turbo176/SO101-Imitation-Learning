"""Synchronize an existing trial's real camera frames and recorded joint positions.

This is a presentation export, not a new rollout or live measurement. The source
Parquet, camera PTS and frame indices must agree before any output is written.
"""
import argparse
import csv
import json
import math
import subprocess
from contextlib import ExitStack
from datetime import datetime
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from build_showcase import ROOT, digest, font, inspect

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
BG, PANEL, GRID = '#f7f7f4', '#eeefeb', '#dedfda'
INK, MUTED, STATE, ACTION = '#252a2b', '#777e7d', '#137f82', '#c18a37'
SIZE = (1440, 900)


@lru_cache(maxsize=32)
def ui_font(size, bold=False):
    try:
        return ImageFont.truetype('seguisb.ttf' if bold else 'segoeui.ttf', size)
    except OSError:
        return font(size, bold)


def dashed(draw, points, fill, width=2):
    """Draw screen-space dashes without resampling recorded position values."""
    distance = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            continue
        offset = 0.0
        while offset < length:
            phase = (distance + offset) % 12
            step = min(length - offset, (7 if phase < 7 else 12) - phase)
            if phase < 7:
                start, end = offset / length, (offset + step) / length
                draw.line((x1 + (x2 - x1) * start, y1 + (y2 - y1) * start,
                           x1 + (x2 - x1) * end, y1 + (y2 - y1) * end), fill=fill, width=width)
            offset += max(step, 1e-8)
        distance += length


def read_source(dataset, trial):
    folder = dataset / trial
    parquet = folder / 'data/chunk-000/episode_000000.parquet'
    meta_path = folder / 'meta/info.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    table = pd.read_parquet(parquet)
    count, fps = len(table), meta['fps']
    times = table['timestamp'].to_numpy(dtype=float)
    expected_names = [name + '.pos' for name in JOINTS]
    for key in ['action', 'observation.state']:
        if meta['features'][key]['names'] != expected_names:
            raise ValueError(f'Unexpected joint order: {key}')
    if fps != 30 or not np.array_equal(table['frame_index'], np.arange(count)):
        raise ValueError('Expected consecutive 30 FPS frame indices.')
    if not np.allclose(times, np.arange(count) / fps, rtol=0, atol=2e-6):
        raise ValueError('Dataset timestamps do not match the recorded frame grid.')
    state = np.stack(table['observation.state']).astype(float)
    action = np.stack(table['action']).astype(float)
    if state.shape != (count, 6) or action.shape != (count, 6):
        raise ValueError('Expected six recorded positions per row.')
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError('Non-finite position data.')
    sources = []
    for path in [parquet, meta_path]:
        sources.append(dict(dataset_relative_path=path.relative_to(dataset).as_posix(),
                            sha256=digest(path), bytes=path.stat().st_size,
                            mtime_local=datetime.fromtimestamp(path.stat().st_mtime).isoformat()))
    videos, checks = [], {}
    for camera in ['fixed', 'handeye']:
        path = folder / f'videos/chunk-000/observation.images.{camera}/episode_000000.mp4'
        with av.open(str(path)) as container:
            pts = np.array([float(frame.time) for frame in container.decode(video=0)])
        if len(pts) > count or not np.allclose(pts, times[:len(pts)], rtol=0, atol=2e-6):
            raise ValueError(f'{camera} video PTS disagree with dataset timestamps.')
        if camera == 'fixed' and len(pts) != count:
            raise ValueError('Fixed-camera and Parquet frame counts disagree.')
        checks[camera] = dict(decoded_frames=len(pts), first_pts_seconds=float(pts[0]),
                              last_pts_seconds=float(pts[-1]),
                              max_timestamp_difference_seconds=float(abs(pts-times[:len(pts)]).max()))
        sources.append(dict(dataset_relative_path=path.relative_to(dataset).as_posix(),
                            sha256=digest(path), **inspect(path)))
        videos.append(path)
    alignment = dict(rows=count, fps=fps, consecutive_frame_indices=True,
                     timestamp_grid_max_error_seconds=float(abs(times-np.arange(count)/fps).max()),
                     cameras=checks, hardware_exposure_synchronization_measured=False)
    return table, times, state, action, videos, sources, alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path('../datasets'))
    parser.add_argument('--trial', choices=['eval_medicine_box_black_050000_trial06'],
                        default='eval_medicine_box_black_050000_trial06')
    parser.add_argument('--stills-dir', type=Path, help='Optional directory for visual-review frames.')
    args = parser.parse_args()
    dataset = args.dataset_root.resolve()
    table, times, state, action, videos, sources, alignment = read_source(dataset, args.trial)
    count, fps = len(table), alignment['fps']
    media = ROOT / 'media/so101'
    result_dir = ROOT / 'results/so101/telemetry'
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / 'trial06.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['frame_index', 'timestamp', *['state.'+j for j in JOINTS],
                         *['action.'+j for j in JOINTS]])
        for i in range(count):
            writer.writerow([i, format(times[i], '.9g'), *[format(v, '.9g') for v in state[i]],
                             *[format(v, '.9g') for v in action[i]]])

    base = Image.new('RGB', SIZE, BG)
    draw = ImageDraw.Draw(base)
    draw.text((32, 20), 'SO-101', font=ui_font(39, True), fill=INK)
    draw.text((191, 32), 'Pick & place', font=ui_font(25), fill=MUTED)
    draw.line((32, 85, 1408, 85), fill=GRID, width=1)
    draw.text((32, 103), 'Fixed', font=ui_font(22), fill=MUTED)
    draw.text((32, 634), 'Wrist', font=ui_font(22), fill=MUTED)
    draw.text((716, 103), 'Joint position', font=ui_font(24, True), fill=INK)
    draw.line((1042, 120, 1073, 120), fill=STATE, width=4)
    draw.text((1083, 102), 'Measured', font=ui_font(22), fill=INK)
    dashed(draw, [(1240, 120), (1271, 120)], ACTION, 3)
    draw.text((1281, 102), 'Target', font=ui_font(22), fill=INK)

    curves = Image.new('RGBA', SIZE, (0, 0, 0, 0))
    line_draw = ImageDraw.Draw(curves)
    x0, x1, chart_top, stride = 772, 1408, 161, 109
    xs = x0 + (x1-x0) * times / (count/fps)
    chart_rows = []
    for joint, name in enumerate(JOINTS):
        top = chart_top+joint*stride
        low, high = (-110, 110) if joint < 5 else (-10, 105)
        y0, y1 = top+37, top+94
        draw.text((716, top), name.replace('_', ' ').capitalize(), font=ui_font(23), fill=INK)
        for value in ([-100, 0, 100] if joint < 5 else [0, 50, 100]):
            y = y1-(value-low)/(high-low)*(y1-y0)
            draw.line((x0, y, x1, y), fill=GRID, width=1)
            if joint in [0, 5]:
                draw.text((x0-12, y), str(value), font=ui_font(18), fill=MUTED, anchor='rm')
        for tick in [0, 5, 10, 15]:
            x = x0+(x1-x0)*tick/(count/fps)
            draw.line((x, y0, x, y1), fill=GRID, width=1)
        sy = y1-(state[:, joint]-low)/(high-low)*(y1-y0)
        ay = y1-(action[:, joint]-low)/(high-low)*(y1-y0)
        line_draw.line(list(zip(xs, sy)), fill=STATE, width=4)
        dashed(line_draw, list(zip(xs, ay)), ACTION, 3)
        chart_rows.append((top, y0, y1, sy, ay))
    for tick in [0, 5, 10, 15]:
        draw.text((x0+(x1-x0)*tick/(count/fps), 838), str(tick), font=ui_font(21), fill=MUTED, anchor='mt')
    draw.text((1408, 838), 's', font=ui_font(21), fill=MUTED, anchor='rt')
    output = media / 'deployment-trial06-telemetry.mp4'
    with ExitStack() as stack:
        containers = [stack.enter_context(av.open(str(path))) for path in videos]
        readers = [iter(c.decode(video=0)) for c in containers]
        out = stack.enter_context(av.open(str(output), 'w', options={'movflags': '+faststart'}))
        stream = out.add_stream('libx264', rate=fps)
        stream.width, stream.height, stream.pix_fmt = *SIZE, 'yuv420p'
        stream.options = {'crf': '19', 'preset': 'medium'}
        for i in range(count):
            canvas = base.copy()
            fixed, wrist = next(readers[0]), next(readers[1], None)
            canvas.paste(fixed.to_image(), (32, 140))
            if wrist is not None:
                canvas.paste(wrist.to_image().resize((288, 216), Image.Resampling.LANCZOS), (32, 672))
            else:
                d = ImageDraw.Draw(canvas)
                d.rectangle((32, 672, 320, 888), fill=PANEL)
                d.text((176, 763), 'Recording ended', font=ui_font(25), fill=INK, anchor='mm')
                d.text((176, 801), '14.20 s', font=ui_font(22), fill=MUTED, anchor='mm')
            right = min(x1+1, int(math.ceil(xs[i]))+1)
            canvas.paste(curves.crop((x0, chart_top, right, 810)), (x0, chart_top),
                         curves.crop((x0, chart_top, right, 810)))
            d = ImageDraw.Draw(canvas)
            d.text((1408, 34), f'{times[i]:05.2f} / {count/fps:.2f} s',
                   font=ui_font(25), fill=MUTED, anchor='ra')
            for j, (top, y0, y1, sy, ay) in enumerate(chart_rows):
                x = float(xs[i])
                d.line((x, y0, x, y1), fill='#919a97', width=1)
                for y, color in [(float(sy[i]), STATE), (float(ay[i]), ACTION)]:
                    d.ellipse((x-4, y-4, x+4, y+4), fill=color)
                d.text((1408, top), f'{state[i,j]:.1f}', font=ui_font(24), fill=STATE, anchor='ra')
            if args.stills_dir and i in [0, 241, 321, 426, 562]:
                args.stills_dir.mkdir(parents=True, exist_ok=True)
                canvas.save(args.stills_dir / f'telemetry-frame-{i:03d}.png')
            frame = av.VideoFrame.from_image(canvas)
            frame.pts, frame.time_base = i, Fraction(1, fps)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)

    import imageio_ffmpeg
    gif = output.with_suffix('.gif')
    filters = ('[0:v]fps=10,scale=960:600:flags=lanczos,split[a][b];'
               '[a]palettegen=max_colors=96:stats_mode=diff[p];'
               '[b][p]paletteuse=dither=bayer:bayer_scale=4')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', str(output), '-filter_complex', filters, '-loop', '0', str(gif)], check=True)
    provenance = dict(trial=args.trial, joint_order=JOINTS, alignment=alignment, sources=sources,
                      position_units='Calibration-normalized servo position: body -100..100, gripper 0..100; not radians or degrees.',
                      plotted_columns={'solid_teal': 'observation.state', 'dashed_amber': 'action'},
                      action_semantics='Recorded action returned by robot.send_action; not offline model predictions.',
                      current_or_torque_recorded=False, time_cuts=False, playback_speed=1,
                      controller='Unverified; 050000 in directory name, no controller or checkpoint hash.',
                      csv_sha256=digest(csv_path), csv_rows=count,
                      camera_transforms='Fixed: native 640x480; wrist: resized to 288x216; no crop or retouching.',
                      presentation=dict(layout='Light neutral canvas; six unboxed position plots; one measured value per joint.',
                                        visible_labels=['SO-101', 'Pick & place', 'Fixed', 'Wrist',
                                                        'Joint position', 'Measured', 'Target', 'elapsed / duration'],
                                        technical_notes='Source, units, recording completeness and controller provenance remain in the companion documentation.'),
                      missing_handeye_frames=count-alignment['cameras']['handeye']['decoded_frames'])
    (result_dir/'trial06.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    manifest_path = ROOT/'results/so101/media_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for path in [output, gif]:
        asset = path.relative_to(ROOT).as_posix()
        row = dict(asset=asset, sha256=digest(path), bytes=path.stat().st_size, sources=sources,
                   transformations='Minimal light layout with real camera playback and six measured/sent-position histories; '
                                   'one measured value per joint; source details in companion docs; no future trace revealed. '
                                   'Entire recorded interval; missing wrist interval replaced by an explicit card.',
                   control_label='Deployment recording; model identity and autonomous-control provenance unverified',
                   playback_speed=1, time_cuts=False, checkpoint_identifier=provenance['controller'],
                   telemetry='results/so101/telemetry/trial06.csv', alignment=alignment,
                   missing_handeye_frames=provenance['missing_handeye_frames'], **inspect(path))
        manifest[:] = [r for r in manifest if r['asset'] != asset]
        manifest.append(row)
        print(f'{asset}: {path.stat().st_size:,} bytes', flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(alignment, indent=2), flush=True)


if __name__ == '__main__':
    main()
