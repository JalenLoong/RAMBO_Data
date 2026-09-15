"""Read-only MP4/PNG validation; no encoder or simulator imports."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from ..contracts_v2.validation import ContractError, require, digest


@dataclass(frozen=True)
class MediaTools:
    ffmpeg: str
    ffprobe: str

    def __post_init__(self):
        for value in (self.ffmpeg, self.ffprobe):
            require(isinstance(value, str) and bool(value), 'config', 'explicit media tool paths required')
            require(Path(value).is_file() and os.access(value, os.X_OK), 'config', f'not executable: {value}')


def run(args, *, binary=False):
    try:
        result = subprocess.run(args, capture_output=True, text=not binary, check=False)
    except OSError as error:
        raise ContractError('media', str(error)) from error
    require(result.returncode == 0, 'media', str(result.stderr)[-2000:])
    return result.stdout


def validate_video(path, *, count, fps, tools, profile=None, receipt=None, allow_partial=False):
    path = Path(path)
    require(path.is_file() and path.suffix.lower() == '.mp4', 'media', 'MP4 file required')
    require(allow_partial or not path.name.endswith('.partial.mp4'), 'media', 'unfinished video')
    require(type(count) is int and count > 0, 'media', 'positive expected frame count')
    try:
        data = json.loads(run([tools.ffprobe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                              '-show_streams', '-show_frames', '-show_entries',
                              'stream=codec_name,width,height,pix_fmt,color_range,color_space,color_transfer,color_primaries,r_frame_rate,avg_frame_rate,nb_read_frames:frame=key_frame,best_effort_timestamp_time',
                              '-of', 'json', str(path)]))
        stream = data['streams'][0]
        frames = data['frames']
        pts = np.asarray([float(frame['best_effort_timestamp_time']) for frame in frames])
        require(int(stream['nb_read_frames']) == count and len(frames) == count, 'media', 'decoded frame count')
        require(Fraction(stream['r_frame_rate']) == fps and Fraction(stream['avg_frame_rate']) == fps,
                'media', 'CFR metadata')
    except (ValueError, KeyError, IndexError, ZeroDivisionError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError('media', f'invalid probe response: {error}') from error
    require(np.allclose(pts, np.arange(count) / fps, rtol=0, atol=1e-6), 'media', 'display-order CFR timestamps/zero origin')
    require(stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p', 'media', 'codec/pixel format')
    if profile is not None:
        require((stream['width'], stream['height']) == (1280, 720), 'media', 'policy resolution')
        for field in ('color_range', 'color_space', 'color_transfer', 'color_primaries'):
            require(stream.get(field) == profile[field], 'media', field)
        keys = [i for i, frame in enumerate(frames) if frame['key_frame']]
        require(keys == list(range(0, count, 50)), 'media', 'GOP/keyframe placement')
        require(receipt is not None and receipt['profile_hash'] == digest(profile), 'media', 'encoding profile receipt')
        command = receipt['command']
        for field, option in [('encoder', '-c:v'), ('crf', '-crf'), ('preset', '-preset'), ('gop', '-g'),
                              ('keyint_min', '-keyint_min'), ('scenecut', '-sc_threshold'), ('color_range', '-color_range')]:
            require(receipt[field] == profile[field], 'media', f'encoding receipt {field}')
            require(command.count(option) == 1 and command.index(option) + 1 < len(command), 'media', f'missing encoding option {option}')
            require(command[command.index(option) + 1] == str(profile[field]), 'media', f'encoding command {field}')
        require(receipt['encoder_version'], 'media', 'encoder provenance')
    run([tools.ffmpeg, '-v', 'error', '-xerror', '-err_detect', 'explode', '-i', str(path), '-f', 'null', '-'])
    return {'frames': count, 'fps': fps, 'stream': stream, 'complete_decode': True}


def decode_frame(path, index, tools):
    require(type(index) is int and index >= 0, 'media', 'frame index')
    raw = run([tools.ffmpeg, '-v', 'error', '-i', str(path), '-vf', f'select=eq(n\\,{index})',
               '-frames:v', '1', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'], binary=True)
    require(len(raw) == 720 * 1280 * 3, 'media', 'decoded RGB8 frame shape')
    return np.frombuffer(raw, np.uint8).reshape(720, 1280, 3).copy()


def read_terminal_png(path):
    try:
        with Path(path).open('rb') as file:
            header = file.read(26)
        require(header[:8] == b'\x89PNG\r\n\x1a\n' and header[24:26] == bytes([8, 2]), 'terminal', 'RGB8 PNG required')
        with Image.open(path) as im:
            require(im.mode == 'RGB' and im.size == (1280, 720), 'terminal', 'terminal image shape')
            return np.asarray(im).copy()
    except (OSError, ValueError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError('terminal', str(error)) from error
