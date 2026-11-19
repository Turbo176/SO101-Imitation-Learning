"""Build the README's labeled real-camera media from existing local recordings.

Requires PyAV, Pillow and imageio-ffmpeg. Sources are read without modification.
All outputs retain real camera geometry; missing footage is shown as a text card.
"""
import argparse
import hashlib
import json
import subprocess
from contextlib import ExitStack
from datetime import datetime
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BG = '#152b3d'
INK = '#18374b'
MUTED = '#587184'


@lru_cache(maxsize=32)
def font(size, bold=False):
    names = ['DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf',
             'arialbd.ttf' if bold else 'arial.ttf']
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    raise RuntimeError('Install DejaVu Sans or Arial for legible media labels.')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        count = sum(1 for _ in container.decode(video=0))
        return dict(decoded_frames=count, fps=str(stream.average_rate),
                    width=stream.width, height=stream.height,
                    duration_seconds=count / float(stream.average_rate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path('../datasets'))
    args = parser.parse_args()
    dataset = args.dataset_root.resolve()
    media = ROOT / 'media/so101'
    manifest_path = ROOT / 'results/so101/media_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    def source(path):
        return dict(dataset_relative_path=path.relative_to(dataset).as_posix(),
                    sha256=digest(path),
                    mtime_local=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    **inspect(path))

    def register(path, sources, description, label, **extra):
        asset = path.relative_to(ROOT).as_posix()
        row = dict(asset=asset, sha256=digest(path), bytes=path.stat().st_size,
                   sources=sources, transformations=description,
                   control_label=label, playback_speed=1, time_cuts=False)
        row.update(extra)
        if path.suffix in ['.mp4', '.gif']:
            row.update(inspect(path))
        manifest[:] = [r for r in manifest if r['asset'] != asset]
        manifest.append(row)
        print(f'{asset}: {path.stat().st_size:,} bytes', flush=True)

    def paths(folder, episode=0):
        return [dataset / folder / 'videos/chunk-000' /
                f'observation.images.{camera}/episode_{episode:06d}.mp4'
                for camera in ['fixed', 'handeye']]

    def pair(name, folder, title, control, episode=0):
        inputs = paths(folder, episode)
        sources = [source(p) for p in inputs]
        count = sources[0]['decoded_frames']
        assert sources[0]['fps'] == sources[1]['fps'] == '30'
        assert sources[1]['decoded_frames'] <= count
        output = media / f'{name}.mp4'
        with ExitStack() as stack:
            containers = [stack.enter_context(av.open(str(p))) for p in inputs]
            readers = [iter(c.decode(video=0)) for c in containers]
            out = stack.enter_context(av.open(str(output), 'w', options={'movflags': '+faststart'}))
            stream = out.add_stream('libx264', rate=30)
            stream.width, stream.height, stream.pix_fmt = 1280, 600, 'yuv420p'
            stream.options = {'crf': '19', 'preset': 'medium'}
            for i in range(count):
                canvas = Image.new('RGB', (1280, 600), BG)
                draw = ImageDraw.Draw(canvas)
                draw.text((18, 9), title, font=font(25, True), fill='white')
                draw.text((1260, 13), f'{i / 30:05.2f} s / {count / 30:.2f} s',
                          font=font(21), fill='#afcfdb', anchor='ra')
                for view, reader in enumerate(readers):
                    frame = next(reader, None)
                    x = view * 640
                    draw.text((x + 18, 44), 'FIXED  /  TABLETOP' if view == 0 else 'HANDEYE  /  WRIST VIEW',
                              font=font(20, True), fill='#6bddc5' if view == 0 else '#a8c7ff')
                    if frame is not None:
                        image = frame.to_image()
                        assert image.size == (640, 480)
                        canvas.paste(image, (x, 76))
                    else:
                        draw.rectangle((x, 76, x + 640, 556), fill='#253d4e')
                        draw.text((x + 320, 280), 'HANDEYE RECORDING ENDED',
                                  font=font(25, True), fill='white', anchor='mm')
                        draw.text((x + 320, 322),
                                  f'No frames after {sources[1]["decoded_frames"] / 30:.2f} s',
                                  font=font(23), fill='#b5cbd5', anchor='mm')
                footer = 'HUMAN TELEOPERATION  |  training demonstration' if episode else 'DEPLOYMENT RECORDING  |  controller / checkpoint hash not logged'
                draw.text((18, 570), footer, font=font(19), fill='#c4d6e0')
                draw.text((1260, 570), '1x playback  |  full interval', font=font(19), fill='#c4d6e0', anchor='ra')
                video_frame = av.VideoFrame.from_image(canvas)
                video_frame.pts, video_frame.time_base = i, Fraction(1, 30)
                for packet in stream.encode(video_frame):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
        checkpoint = '050000 in source directory name; controller/hash not recorded' if not episode else None
        description = ('Native 640x480 views arranged left=fixed, right=handeye; labels added outside camera images; '
                       'frame-index alignment from the same dataset episode, no exposure-sync claim; entire fixed-camera interval.')
        missing = count - sources[1]['decoded_frames']
        if missing:
            description += f' Final {missing} missing handeye frames shown as an explicit text card, never frozen or generated camera footage.'
        register(output, sources, description, control,
                 checkpoint_identifier=checkpoint, missing_handeye_frames=missing,
                 encoding={'codec': 'libx264', 'crf': 19, 'fps': 30})
        import imageio_ffmpeg
        gif = output.with_suffix('.gif')
        filters = ('[0:v]fps=10,scale=768:360:flags=lanczos,split[a][b];'
                   '[a]palettegen=max_colors=96:stats_mode=diff[p];'
                   '[b][p]paletteuse=dither=bayer:bayer_scale=4')
        command = ['-filter_complex', filters, '-loop', '0']
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                        '-y', '-i', str(output), *command, str(gif)], check=True)
        register(gif, sources, '768x360 / 10 FPS preview of the labeled full-interval two-view MP4; 1x. ' + description,
                 control, checkpoint_identifier=checkpoint, missing_handeye_frames=missing,
                 ffmpeg_output_arguments=command)
        return inputs

    deployment = 'Deployment recording; model identity and autonomous-control provenance unverified'
    hero = pair('deployment-trial06-two-view', 'eval_medicine_box_black_050000_trial06',
                'TRIAL 06  /  REAL-ROBOT PICK AND PLACE', deployment)
    pair('deployment-trial04-two-view', 'eval_medicine_box_black_050000_trial04',
         'TRIAL 04  /  GRASP, PLACE AND HUMAN RESET', deployment + '; human cable handling and reset retained')
    demo = pair('teleoperation-episode54', 'medicine_box_black_5_20s_v1',
                'EPISODE 54  /  HUMAN DEMONSTRATION', 'Human teleoperation demonstration; training set', episode=54)
    pair('deployment-trial03-two-view', 'eval_medicine_box_black_050000_trial03',
         'TRIAL 03  /  BOX TIPS, NO COMPLETED PLACEMENT', deployment + '; failure observation, human hand at end')

    def frames_at(path, indices):
        found = {}
        with av.open(str(path)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i in indices:
                    found[i] = frame.to_image()
                if i >= max(indices):
                    break
        assert set(found) == set(indices)
        return found

    indices = [241, 321, 401, 482]
    frames = frames_at(hero[0], indices)
    image = Image.new('RGB', (1360, 388), '#f0f5f8')
    draw = ImageDraw.Draw(image)
    draw.text((20, 15), 'ONE RECORDED TRAJECTORY', fill=INK, font=font(24, True))
    draw.text((20, 49), 'Trial06 / fixed camera / selected source frames / controller identity unverified', fill=MUTED, font=font(19))
    for col, (idx, label) in enumerate(zip(indices, ['Approach', 'Lift', 'Lower to target', 'Released'])):
        x = 20 + col * 335
        image.paste(frames[idx].resize((315, 236), Image.Resampling.LANCZOS), (x, 88))
        draw.text((x, 338), f'{col + 1:02d}  {label}', font=font(22, True), fill=INK)
        draw.text((x, 367), f't = {idx / 30:.2f} s', font=font(17), fill=MUTED)
    output = media / 'task-sequence.webp'
    image.save(output, quality=92, method=6)
    register(output, [source(hero[0])], 'Four selected source frames resized for a storyboard; no retouching or synthesized views.',
             deployment, selected_source_frame_indices=indices, time_cuts=True)

    image = Image.new('RGB', (1320, 620), '#f0f5f8')
    draw = ImageDraw.Draw(image)
    draw.text((20, 14), 'TWO COMPLEMENTARY CAMERA INPUTS', font=font(27, True), fill=INK)
    for col, path in enumerate(demo):
        x = 20 + col * 650
        draw.text((x, 61), 'FIXED  /  scene and target area' if not col else 'HANDEYE  /  close view of object and gripper',
                  font=font(22, True), fill=INK)
        image.paste(frames_at(path, [210])[210], (x, 98))
    draw.text((20, 592), 'Same recorded episode and frame index: demonstration 54 / frame 210 / 7.00 s / native camera geometry',
              font=font(20), fill=MUTED)
    output = media / 'camera-inputs.webp'
    image.save(output, quality=92, method=6)
    register(output, [source(p) for p in demo], 'One real frame per camera, matched by episode and frame index; labels outside images.',
             'Human teleoperation demonstration; training set', selected_source_frame_indices=[210, 210])
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
