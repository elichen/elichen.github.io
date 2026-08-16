#!/usr/bin/env python3
"""Transcode a video into LO-FI LOL's compact, delta-coded ASCII frame format."""

from __future__ import annotations

import argparse
import struct
import subprocess
from pathlib import Path


GLYPHS = " .,:;irsXA253hMHGS#9B&@|-\\"
BAYER_4 = (
    0, 8, 2, 10,
    12, 4, 14, 6,
    3, 11, 1, 9,
    15, 7, 13, 5,
)
HEADER = struct.Struct("<4sBBBBI")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def encode_delta(current: list[int], previous: list[int]) -> bytes:
    encoded = bytearray()
    index = 0
    total = len(current)
    while index < total:
        unchanged = current[index] == previous[index]
        run = 1
        while run < 128 and index + run < total and (current[index + run] == previous[index + run]) == unchanged:
            run += 1
        if unchanged:
            encoded.append(0x80 | (run - 1))
        else:
            encoded.append(run - 1)
            for value in current[index:index + run]:
                encoded.extend(struct.pack("<H", value))
        index += run
    return bytes(encoded)


def transcode(source: Path, destination: Path, columns: int, rows: int, fps: int) -> int:
    sample_width = columns * 2
    sample_height = rows * 2
    frame_bytes = sample_width * sample_height * 3
    # Stretching into the sample grid compensates for the roughly 0.58:1
    # width-to-height ratio of a monospace glyph when the player draws it.
    filter_graph = f"scale={sample_width}:{sample_height}:flags=lanczos,fps={fps},format=rgb24"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-vf", filter_graph, "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("ffmpeg did not expose a video stream")

    destination.parent.mkdir(parents=True, exist_ok=True)
    previous = [0] * (columns * rows)
    previous_levels = [-1] * (columns * rows)
    exposure_low = None
    exposure_high = None
    frame_count = 0

    with destination.open("wb") as output:
        output.write(HEADER.pack(b"ASCV", 1, columns, rows, fps, 0))
        while True:
            raw = process.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError(f"truncated raw frame in {source}")

            lumas: list[float] = []
            colors: list[tuple[float, float, float]] = []
            gradients: list[tuple[float, float]] = []
            for y in range(rows):
                for x in range(columns):
                    offsets = (
                        ((y * 2) * sample_width + x * 2) * 3,
                        ((y * 2) * sample_width + x * 2 + 1) * 3,
                        (((y * 2) + 1) * sample_width + x * 2) * 3,
                        (((y * 2) + 1) * sample_width + x * 2 + 1) * 3,
                    )
                    samples = [raw[offset:offset + 3] for offset in offsets]
                    cell_lumas = [pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722 for pixel in samples]
                    lumas.append(sum(cell_lumas) / 4)
                    colors.append(tuple(sum(pixel[channel] for pixel in samples) / 4 for channel in range(3)))
                    gradients.append((
                        cell_lumas[1] + cell_lumas[3] - cell_lumas[0] - cell_lumas[2],
                        cell_lumas[2] + cell_lumas[3] - cell_lumas[0] - cell_lumas[1],
                    ))

            low = percentile(lumas, 0.035)
            high = percentile(lumas, 0.965)
            exposure_low = low if exposure_low is None else exposure_low * 0.84 + low * 0.16
            exposure_high = high if exposure_high is None else exposure_high * 0.84 + high * 0.16
            if exposure_high - exposure_low < 48:
                exposure_high = exposure_low + 48

            cells: list[int] = []
            density_glyphs = len(GLYPHS) - 4
            for cell, (luma, color, gradient) in enumerate(zip(lumas, colors, gradients)):
                x = cell % columns
                y = cell // columns
                dither = ((BAYER_4[(y % 4) * 4 + (x % 4)] + 0.5) / 16 - 0.5) * 0.13
                light = clamp((luma - exposure_low) / (exposure_high - exposure_low) + dither, 0, 1)
                level = round(light * (density_glyphs - 1))
                gx, gy = gradient
                edge_strength = (gx * gx + gy * gy) ** 0.5
                glyph = level
                if edge_strength > 72 and light > 0.12:
                    if abs(gx) > abs(gy) * 1.8:
                        glyph = density_glyphs
                    elif abs(gy) > abs(gx) * 1.8:
                        glyph = density_glyphs + 1
                    else:
                        glyph = density_glyphs + (2 if gx * gy > 0 else 3)
                elif previous_levels[cell] >= 0 and abs(previous_levels[cell] - level) <= 1:
                    glyph = previous_levels[cell]
                    level = glyph
                previous_levels[cell] = level

                red, green, blue = color
                red_level = round(clamp(red * 1.08 + 16, 0, 255) / 51)
                green_level = round(clamp(green * 1.08 + 16, 0, 255) / 51)
                blue_level = round(clamp(blue * 1.08 + 16, 0, 255) / 51)
                palette_index = red_level * 36 + green_level * 6 + blue_level
                cells.append(glyph | (palette_index << 8))

            delta = encode_delta(cells, previous)
            output.write(struct.pack("<I", len(delta)))
            output.write(delta)
            previous = cells
            frame_count += 1

        output.seek(0)
        output.write(HEADER.pack(b"ASCV", 1, columns, rows, fps, frame_count))

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--columns", type=int, default=96)
    parser.add_argument("--rows", type=int, default=31)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    frames = transcode(args.source, args.destination, args.columns, args.rows, args.fps)
    print(f"{args.destination}: {frames} frames at {args.columns}x{args.rows} / {args.fps} fps")


if __name__ == "__main__":
    main()
