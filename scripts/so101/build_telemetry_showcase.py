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
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from build_showcase import ROOT, digest, font, inspect

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
BG, PANEL, GRID = '#102838', '#173747', '#365260'
INK, MUTED, STATE, ACTION = '#edf6fa', '#aec7d2', '#55dfbe', '#ffbd72'
SIZE = (1440, 936)


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
    draw.text((22, 17), 'REAL ROBOT / RECORDED JOINT MOTION', font=font(31, True), fill=INK)
    draw.text((22, 62), 'Trial06  |  synchronized playback  |  1x  |  full 18.77 s', font=font(23), fill=MUTED)
    draw.text((22, 111), 'FIXED / TABLETOP', font=font(23, True), fill=STATE)
    draw.text((22, 635), 'HANDEYE / WRIST', font=font(21, True), fill=STATE)
    draw.text((348, 680), 'ONE RECORDED TRAJECTORY', font=font(18, True), fill=INK)
    for line, text in enumerate(['563 control samples / 30 FPS', 'Position units: normalized servo', 'Body: -100..100 / gripper: 0..100',
                                 'No current or torque recorded.', 'Controller / checkpoint hash:', 'not retained in this trial.']):
        draw.text((348, 715+line*25), text, font=font(16), fill=MUTED)
    draw.line((704, 110, 737, 110), fill=STATE, width=4)
    draw.text((749, 96), 'Measured position', font=font(22), fill=INK)
    dashed(draw, [(1011, 110), (1052, 110)], ACTION, 3)
    draw.text((1067, 96), 'Sent target', font=font(22), fill=INK)
    draw.text((704, 133), 'Position history / recorded units / shared time axis', font=font(18), fill=MUTED)

    curves = Image.new('RGBA', SIZE, (0, 0, 0, 0))
    line_draw = ImageDraw.Draw(curves)
    x0, x1, chart_top, stride = 755, 1405, 165, 111
    xs = x0 + (x1-x0) * times / (count/fps)
    chart_rows = []
    for joint, name in enumerate(JOINTS):
        top = chart_top+joint*stride
        low, high = (-110, 110) if joint < 5 else (-10, 105)
        y0, y1 = top+31, top+100
        draw.rounded_rectangle((692, top, 1422, top+105), radius=9, fill=PANEL)
        draw.text((705, top+2), name.replace('_', ' '), font=font(24, True), fill=INK)
        for value in ([-100, 0, 100] if joint < 5 else [0, 50, 100]):
            y = y1-(value-low)/(high-low)*(y1-y0)
            draw.line((x0, y, x1, y), fill=GRID, width=1)
            draw.text((x0-9, y), str(value), font=font(14), fill=MUTED, anchor='rm')
        for tick in [0, 5, 10, 15]:
            x = x0+(x1-x0)*tick/(count/fps)
            draw.line((x, y0, x, y1), fill=GRID, width=1)
        sy = y1-(state[:, joint]-low)/(high-low)*(y1-y0)
        ay = y1-(action[:, joint]-low)/(high-low)*(y1-y0)
        line_draw.line(list(zip(xs, sy)), fill=STATE, width=3)
        dashed(line_draw, list(zip(xs, ay)), ACTION, 2)
        chart_rows.append((top, y0, y1, sy, ay))
    for tick in [0, 5, 10, 15]:
        draw.text((x0+(x1-x0)*tick/(count/fps), 841), f'{tick} s', font=font(18), fill=MUTED, anchor='mt')
    draw.text((1405, 841), '18.77 s', font=font(18), fill=MUTED, anchor='rt')
    draw.text((22, 914), 'Recorded frames + Parquet samples; paired by timestamps / frame index. Hardware exposure sync not measured.',
              font=font(16), fill=MUTED)
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
            canvas.paste(fixed.to_image(), (22, 145))
            if wrist is not None:
                canvas.paste(wrist.to_image().resize((304, 228), Image.Resampling.LANCZOS), (22, 665))
            else:
                d = ImageDraw.Draw(canvas)
                d.rectangle((22, 665, 326, 893), fill=PANEL)
                d.text((174, 756), 'HANDEYE ENDED', font=font(24, True), fill=INK, anchor='mm')
                d.text((174, 795), 'No frames after 14.20 s', font=font(18), fill=MUTED, anchor='mm')
            right = min(x1+1, int(math.ceil(xs[i]))+1)
            canvas.paste(curves.crop((x0, chart_top, right, 830)), (x0, chart_top),
                         curves.crop((x0, chart_top, right, 830)))
            d = ImageDraw.Draw(canvas)
            d.text((1416, 57), f'{times[i]:05.2f} s  /  frame {i:03d}', font=font(22), fill=INK, anchor='ra')
            for j, (top, y0, y1, sy, ay) in enumerate(chart_rows):
                x = float(xs[i])
                d.line((x, y0, x, y1), fill='#edf6fa', width=1)
                for y, color in [(float(sy[i]), STATE), (float(ay[i]), ACTION)]:
                    d.ellipse((x-4, y-4, x+4, y+4), fill=color)
                d.text((1250, top+4), f'{state[i,j]:6.1f}', font=font(20, True), fill=STATE, anchor='ra')
                d.text((1407, top+4), f'{action[i,j]:6.1f}', font=font(20, True), fill=ACTION, anchor='ra')
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
    filters = ('[0:v]fps=10,scale=960:624:flags=lanczos,split[a][b];'
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
                      camera_transforms='Fixed: native 640x480; wrist: resized to 304x228; no crop or retouching.',
                      missing_handeye_frames=count-alignment['cameras']['handeye']['decoded_frames'])
    (result_dir/'trial06.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    manifest_path = ROOT/'results/so101/media_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for path in [output, gif]:
        asset = path.relative_to(ROOT).as_posix()
        row = dict(asset=asset, sha256=digest(path), bytes=path.stat().st_size, sources=sources,
                   transformations='Synchronized real camera playback and six measured/sent-position histories; no future trace revealed. '
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
