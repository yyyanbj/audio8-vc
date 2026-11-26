#!/usr/bin/env python3
"""Generate crisp multi-resolution Windows `.ico` and `.png` icons for Audio8-VC.

Uses only the Python standard library (`math`, `struct`, `zlib`) so it can run in any
CI or local build environment without Pillow or Inkscape installed.
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def _clamp_u8(val: float) -> int:
    return max(0, min(255, int(round(val))))


def _blend_over(
    dst: tuple[float, float, float, float],
    src: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    sr, sg, sb, sa = src
    dr, dg, db, da = dst
    out_a = sa + da * (1.0 - sa)
    if out_a <= 1e-6:
        return (0.0, 0.0, 0.0, 0.0)
    out_r = (sr * sa + dr * da * (1.0 - sa)) / out_a
    out_g = (sg * sa + dg * da * (1.0 - sa)) / out_a
    out_b = (sb * sa + db * da * (1.0 - sa)) / out_a
    return (out_r, out_g, out_b, out_a)


def _sdf_rounded_box(px: float, py: float, bx: float, by: float, r: float) -> float:
    qx = abs(px) - bx + r
    qy = abs(py) - by + r
    return math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - r


def _render_rgba(size: int) -> bytes:
    """Render an anti-aliased `size x size` RGBA icon (deep navy rounded tile + voice wave)."""
    out = bytearray(size * size * 4)
    aa = 2.0 / size  # smooth 1px edge width in [-1, 1] normalized coordinates

    # 5 symmetric capsule waveform bars: (center_x, half_height)
    bars = (
        (-0.46, 0.16),
        (-0.23, 0.34),
        (0.00, 0.48),
        (0.23, 0.31),
        (0.46, 0.18),
    )
    bar_half_w = 0.072

    idx = 0
    for y in range(size):
        ny = (2.0 * (y + 0.5) / size) - 1.0  # -1 (top) .. +1 (bottom)
        for x in range(size):
            nx = (2.0 * (x + 0.5) / size) - 1.0  # -1 (left) .. +1 (right)

            # Outer rounded square background
            d_tile = _sdf_rounded_box(nx, ny, 0.90, 0.90, 0.24)
            tile_alpha = max(0.0, min(1.0, 0.5 - d_tile / aa))
            if tile_alpha <= 0.0:
                idx += 4
                continue

            # Diagonal deep-indigo to electric-blue gradient
            t = max(0.0, min(1.0, (nx + ny + 2.0) * 0.25))
            bg_r = 16.0 + t * 18.0
            bg_g = 26.0 + t * 44.0
            bg_b = 58.0 + t * 92.0

            # Soft radial highlight in upper-left
            glow_dist = math.hypot(nx + 0.35, ny + 0.35)
            glow = max(0.0, 1.0 - glow_dist / 1.15) ** 2
            bg_r += glow * 32.0
            bg_g += glow * 58.0
            bg_b += glow * 95.0

            pixel = (bg_r, bg_g, bg_b, tile_alpha)

            # Subtle inner border ring
            ring_dist = abs(d_tile + 0.045)
            ring_alpha = max(0.0, min(1.0, (0.018 - ring_dist) / aa)) * 0.28 * tile_alpha
            if ring_alpha > 0.0:
                pixel = _blend_over(pixel, (130.0, 190.0, 255.0, ring_alpha))

            # Glowing 5-bar voice waveform capsules
            for bx, half_h in bars:
                d_bar = _sdf_rounded_box(nx - bx, ny, bar_half_w, half_h, bar_half_w)
                bar_alpha = max(0.0, min(1.0, 0.5 - d_bar / aa))
                if bar_alpha > 0.0:
                    # Bright cyan-to-white vertical sheen across waveform bars
                    gt = max(0.0, min(1.0, (ny + 0.55) / 1.1))
                    r = 232.0 - gt * 120.0
                    g = 248.0 - gt * 36.0
                    b = 255.0 - gt * 4.0
                    pixel = _blend_over(pixel, (r, g, b, bar_alpha))

            out[idx] = _clamp_u8(pixel[0])
            out[idx + 1] = _clamp_u8(pixel[1])
            out[idx + 2] = _clamp_u8(pixel[2])
            out[idx + 3] = _clamp_u8(pixel[3])
            idx += 4

    return bytes(out)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def encode_png(size: int, rgba: bytes) -> bytes:
    """Encode an RGBA buffer into a standard PNG byte stream."""
    raw = bytearray()
    stride = size * 4
    for y in range(size):
        raw.append(0)  # filter type 0 (None)
        raw.extend(rgba[y * stride : (y + 1) * stride])

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(raw), level=9)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def _encode_ico_dib(size: int, rgba: bytes) -> bytes:
    """Encode a 32-bit BGRA bottom-up DIB + AND mask for maximum Windows ICO compatibility."""
    # BITMAPINFOHEADER (40 bytes): height is 2*size (XOR + AND mask)
    xor_bytes = bytearray(size * size * 4)
    stride = size * 4
    for y in range(size):
        src_row = (size - 1 - y) * stride
        dst_row = y * stride
        for x in range(size):
            si = src_row + x * 4
            di = dst_row + x * 4
            xor_bytes[di] = rgba[si + 2]      # B
            xor_bytes[di + 1] = rgba[si + 1]  # G
            xor_bytes[di + 2] = rgba[si]      # R
            xor_bytes[di + 3] = rgba[si + 3]  # A

    # 1-bit AND mask padded to 32-bit rows (0 = opaque/use alpha, 1 = transparent)
    and_row_bytes = ((size + 31) // 32) * 4
    and_bytes = bytearray(and_row_bytes * size)
    for y in range(size):
        src_row = (size - 1 - y) * stride
        for x in range(size):
            if rgba[src_row + x * 4 + 3] < 16:
                and_bytes[y * and_row_bytes + (x // 8)] |= 0x80 >> (x % 8)

    header = struct.pack(
        "<IIIHHIIIIII",
        40,                 # biSize
        size,               # biWidth
        size * 2,           # biHeight (XOR + AND)
        1,                  # biPlanes
        32,                 # biBitCount
        0,                  # biCompression (BI_RGB)
        len(xor_bytes) + len(and_bytes),  # biSizeImage
        0,                  # biXPelsPerMeter
        0,                  # biYPelsPerMeter
        0,                  # biClrUsed
        0,                  # biClrImportant
    )
    return header + bytes(xor_bytes) + bytes(and_bytes)


def generate_icons(output_dir: Path) -> tuple[Path, Path]:
    """Generate `audio8_vc.ico` and `audio8_vc.png` inside `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ico_path = output_dir / "audio8_vc.ico"
    png_path = output_dir / "audio8_vc.png"

    entries: list[tuple[int, bytes]] = []
    png_256 = b""
    for size in ICO_SIZES:
        rgba = _render_rgba(size)
        if size == 256:
            png_256 = encode_png(size, rgba)
            entries.append((size, png_256))
        else:
            entries.append((size, _encode_ico_dib(size, rgba)))

    # Write 256x256 PNG
    png_path.write_bytes(png_256)

    # Assemble ICONDIR + ICONDIRENTRY table + payloads
    count = len(entries)
    ico_header = struct.pack("<HHH", 0, 1, count)
    offset = 6 + count * 16
    dir_entries = bytearray()
    payloads = bytearray()
    for size, data in entries:
        dim = 0 if size >= 256 else size
        dir_entries.extend(
            struct.pack(
                "<BBBBHHII",
                dim,
                dim,
                0,
                0,
                1,
                32,
                len(data),
                offset,
            )
        )
        payloads.extend(data)
        offset += len(data)

    ico_path.write_bytes(ico_header + bytes(dir_entries) + bytes(payloads))
    return ico_path, png_path


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "assets"
    ico_path, png_path = generate_icons(out_dir)
    print(f"Generated {ico_path} ({ico_path.stat().st_size} bytes)")
    print(f"Generated {png_path} ({png_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
