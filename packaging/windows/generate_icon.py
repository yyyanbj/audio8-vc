#!/usr/bin/env python3
"""Generate crisp multi-resolution Windows `.ico`, `.png`, and `.svg` icons for Audio8-VC.

Visual identity:
  - Dark obsidian-indigo squircle tile with subtle studio glass rim highlight
  - Sculpted geometric "8" emblem with cyan -> azure -> violet-magenta voice-morph gradient
  - Flanking symmetrical acoustic waveform bars (cyan input left -> magenta-violet output right)

Uses only the Python standard library (`math`, `struct`, `zlib`) so it runs cleanly in any
CI or local build environment without Pillow or Inkscape installed.
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)

ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">
  <defs>
    <linearGradient id="bgGrad" x1="12%" y1="8%" x2="88%" y2="92%">
      <stop offset="0%" stop-color="#111a33"/>
      <stop offset="52%" stop-color="#182246"/>
      <stop offset="100%" stop-color="#261746"/>
    </linearGradient>
    <radialGradient id="cyanGlow" cx="26%" cy="24%" r="58%">
      <stop offset="0%" stop-color="#38e8ff" stop-opacity="0.30"/>
      <stop offset="100%" stop-color="#38e8ff" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="violetGlow" cx="76%" cy="78%" r="58%">
      <stop offset="0%" stop-color="#e056fd" stop-opacity="0.30"/>
      <stop offset="100%" stop-color="#e056fd" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="rimGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#8ae8ff" stop-opacity="0.55"/>
      <stop offset="50%" stop-color="#6b8cff" stop-opacity="0.18"/>
      <stop offset="100%" stop-color="#f08cff" stop-opacity="0.45"/>
    </linearGradient>
    <linearGradient id="eightGrad" x1="18%" y1="12%" x2="84%" y2="88%">
      <stop offset="0%" stop-color="#4df5ff"/>
      <stop offset="46%" stop-color="#5c8dff"/>
      <stop offset="100%" stop-color="#ee6bff"/>
    </linearGradient>
    <linearGradient id="leftWaveGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#88fbff"/>
      <stop offset="100%" stop-color="#36c2ff"/>
    </linearGradient>
    <linearGradient id="rightWaveGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#f794ff"/>
      <stop offset="100%" stop-color="#a855f7"/>
    </linearGradient>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="5" stdDeviation="6" flood-color="#050814" flood-opacity="0.55"/>
    </filter>
  </defs>

  <!-- Rounded Studio Tile -->
  <rect x="14" y="14" width="228" height="228" rx="54" fill="url(#bgGrad)"/>
  <rect x="14" y="14" width="228" height="228" rx="54" fill="url(#cyanGlow)"/>
  <rect x="14" y="14" width="228" height="228" rx="54" fill="url(#violetGlow)"/>
  <rect x="16.5" y="16.5" width="223" height="223" rx="51.5" fill="none" stroke="url(#rimGrad)" stroke-width="3"/>

  <!-- Voice-Morph Emblem: Left input waveform -> Central Audio8 "8" -> Right output waveform -->
  <g filter="url(#softShadow)">
    <!-- Left input soundwave bars (cyan) -->
    <rect x="37" y="110" width="13" height="36" rx="6.5" fill="url(#leftWaveGrad)" opacity="0.88"/>
    <rect x="57" y="92" width="14" height="72" rx="7" fill="url(#leftWaveGrad)"/>

    <!-- Right morphed soundwave bars (violet-magenta) -->
    <rect x="185" y="92" width="14" height="72" rx="7" fill="url(#rightWaveGrad)"/>
    <rect x="206" y="110" width="13" height="36" rx="6.5" fill="url(#rightWaveGrad)" opacity="0.88"/>

    <!-- Sculpted Geometric "8" (dual acoustic rings) -->
    <path
      fill-rule="evenodd"
      fill="url(#eightGrad)"
      d="M 128 54
         A 37 37 0 1 0 128 128
         A 37 37 0 1 0 128 54 Z
         M 128 73.5
         A 17.5 17.5 0 1 1 128 108.5
         A 17.5 17.5 0 1 1 128 73.5 Z"
    />
    <path
      fill-rule="evenodd"
      fill="url(#eightGrad)"
      d="M 128 117
         A 43 43 0 1 0 128 203
         A 43 43 0 1 0 128 117 Z
         M 128 138
         A 22 22 0 1 1 128 182
         A 22 22 0 1 1 128 138 Z"
    />
    <!-- Luminous acoustic nexus dot inside upper & lower chambers -->
    <circle cx="128" cy="91" r="6" fill="#d8fbff" opacity="0.92"/>
    <circle cx="128" cy="160" r="7.5" fill="#ffd9ff" opacity="0.92"/>
  </g>
</svg>
"""


def _clamp_u8(val: float) -> int:
    return max(0, min(255, int(round(val))))


def _clamp01(val: float) -> float:
    return 0.0 if val <= 0.0 else (1.0 if val >= 1.0 else val)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_rgb(
    c1: tuple[float, float, float],
    c2: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    t = _clamp01(t)
    return (
        _lerp(c1[0], c2[0], t),
        _lerp(c1[1], c2[1], t),
        _lerp(c1[2], c2[2], t),
    )


def _blend_over(
    dst: tuple[float, float, float, float],
    src: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Porter-Duff source-over compositing with RGB in [0, 255] and alpha in [0, 1]."""
    sr, sg, sb, sa = src
    dr, dg, db, da = dst
    out_a = sa + da * (1.0 - sa)
    if out_a <= 1e-6:
        return (0.0, 0.0, 0.0, 0.0)
    inv = 1.0 - sa
    out_r = (sr * sa + dr * da * inv) / out_a
    out_g = (sg * sa + dg * da * inv) / out_a
    out_b = (sb * sa + db * da * inv) / out_a
    return (out_r, out_g, out_b, out_a)


def _sdf_rounded_box(px: float, py: float, bx: float, by: float, r: float) -> float:
    qx = abs(px) - bx + r
    qy = abs(py) - by + r
    return math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - r


def _sdf_ring(px: float, py: float, cx: float, cy: float, r_outer: float, r_inner: float) -> float:
    """Signed distance to an annulus (negative inside the ring)."""
    d = math.hypot(px - cx, py - cy)
    return max(d - r_outer, r_inner - d)


def _sdf_circle(px: float, py: float, cx: float, cy: float, r: float) -> float:
    return math.hypot(px - cx, py - cy) - r


def _sample_icon(nx: float, ny: float, aa: float) -> tuple[float, float, float, float]:
    """Evaluate icon RGBA (RGB in 0..255, A in 0..1) at normalized coords [-1, 1]^2.

    Coordinate mapping matches `ICON_SVG` (`viewBox="0 0 256 256"` -> `coord = (px / 128) - 1`).
    """
    # 1. Outer rounded studio tile: x,y in [14, 242] -> half-extent 114/128 = 0.890625, r = 54/128 = 0.421875
    d_tile = _sdf_rounded_box(nx, ny, 0.890625, 0.890625, 0.421875)
    tile_alpha = _clamp01(0.5 - d_tile / aa)
    if tile_alpha <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)

    # Diagonal obsidian-indigo to deep-violet backdrop
    t_bg = _clamp01((nx * 0.76 + ny * 0.84 + 1.6) / 3.2)
    if t_bg < 0.52:
        bg_r, bg_g, bg_b = _lerp_rgb((17.0, 26.0, 51.0), (24.0, 34.0, 70.0), t_bg / 0.52)
    else:
        bg_r, bg_g, bg_b = _lerp_rgb(
            (24.0, 34.0, 70.0), (38.0, 23.0, 70.0), (t_bg - 0.52) / 0.48
        )

    # Soft dual ambient studio glows (cyan top-left, violet bottom-right)
    d_cyan = math.hypot(nx + 0.48, ny + 0.52)
    g_cyan = _clamp01(1.0 - d_cyan / 1.18) ** 2 * 0.34
    d_violet = math.hypot(nx - 0.52, ny - 0.56)
    g_violet = _clamp01(1.0 - d_violet / 1.18) ** 2 * 0.34

    pixel = (bg_r, bg_g, bg_b, tile_alpha)
    if g_cyan > 0.001:
        pixel = _blend_over(pixel, (56.0, 232.0, 255.0, g_cyan * tile_alpha))
    if g_violet > 0.001:
        pixel = _blend_over(pixel, (224.0, 86.0, 253.0, g_violet * tile_alpha))

    # Crisp glass inner rim highlight
    rim_dist = abs(d_tile + 0.024) - 0.012
    rim_alpha = _clamp01(0.5 - rim_dist / aa) * tile_alpha
    if rim_alpha > 0.001:
        t_rim = _clamp01((nx + ny + 2.0) * 0.25)
        if t_rim < 0.5:
            rr, rg, rb = _lerp_rgb((138.0, 232.0, 255.0), (107.0, 140.0, 255.0), t_rim * 2.0)
            ra = _lerp(0.55, 0.20, t_rim * 2.0)
        else:
            rr, rg, rb = _lerp_rgb(
                (107.0, 140.0, 255.0), (240.0, 140.0, 255.0), (t_rim - 0.5) * 2.0
            )
            ra = _lerp(0.20, 0.46, (t_rim - 0.5) * 2.0)
        pixel = _blend_over(pixel, (rr, rg, rb, rim_alpha * ra))

    # Helper SDF for the entire foreground emblem (for soft drop shadow + bloom)
    # Upper ring of "8": center (0, -0.2890625), R_out = 37/128 = 0.2890625, R_in = 17.5/128 = 0.13671875
    # Lower ring of "8": center (0, +0.25), R_out = 43/128 = 0.3359375, R_in = 22/128 = 0.171875
    def _emblem_sdf(x: float, y: float) -> tuple[float, float, float, float]:
        d_top = _sdf_ring(x, y, 0.0, -0.2890625, 0.2890625, 0.13671875)
        d_bot = _sdf_ring(x, y, 0.0, 0.25, 0.3359375, 0.171875)
        d_eight = min(d_top, d_bot)

        # Left waveform bars: x=-0.66015625 (h=0.140625, w=0.05078125), x=-0.50 (h=0.28125, w=0.0546875)
        d_l1 = _sdf_rounded_box(x + 0.66015625, y, 0.05078125, 0.140625, 0.05078125)
        d_l2 = _sdf_rounded_box(x + 0.50, y, 0.0546875, 0.28125, 0.0546875)
        d_left = min(d_l1, d_l2)

        # Right waveform bars (mirror)
        d_r1 = _sdf_rounded_box(x - 0.50, y, 0.0546875, 0.28125, 0.0546875)
        d_r2 = _sdf_rounded_box(x - 0.66015625, y, 0.05078125, 0.140625, 0.05078125)
        d_right = min(d_r1, d_r2)

        return d_eight, d_left, d_right, min(d_eight, d_left, d_right)

    # 2. Soft drop shadow beneath foreground emblem
    _, _, _, d_sh = _emblem_sdf(nx, ny - 0.038)
    if d_sh < 0.10:
        sh_alpha = _clamp01(1.0 - max(0.0, d_sh) / 0.095) ** 2 * 0.48 * tile_alpha
        if sh_alpha > 0.001:
            pixel = _blend_over(pixel, (5.0, 8.0, 20.0, sh_alpha))

    # 3. Foreground shapes
    d_eight, d_left, d_right, _ = _emblem_sdf(nx, ny)

    # Subtle colored neon halo right around the "8"
    if 0.0 < d_eight < 0.11:
        halo = _clamp01(1.0 - d_eight / 0.11) ** 2 * 0.20 * tile_alpha
        t_h = _clamp01((nx * 0.66 + ny * 0.76 + 1.0) * 0.5)
        hr, hg, hb = _lerp_rgb((77.0, 245.0, 255.0), (238.0, 107.0, 255.0), t_h)
        pixel = _blend_over(pixel, (hr, hg, hb, halo))

    # Left input waveform bars (cyan gradient)
    a_left = _clamp01(0.5 - d_left / aa)
    if a_left > 0.001:
        ty = _clamp01((ny + 0.30) / 0.60)
        lr, lg, lb = _lerp_rgb((136.0, 251.0, 255.0), (54.0, 194.0, 255.0), ty)
        # Outer bar slightly softer opacity
        outer_fade = 0.88 if nx < -0.58 else 1.0
        pixel = _blend_over(pixel, (lr, lg, lb, a_left * outer_fade))

    # Right output waveform bars (violet-magenta gradient)
    a_right = _clamp01(0.5 - d_right / aa)
    if a_right > 0.001:
        ty = _clamp01((ny + 0.30) / 0.60)
        rr, rg, rb = _lerp_rgb((247.0, 148.0, 255.0), (168.0, 85.0, 247.0), ty)
        outer_fade = 0.88 if nx > 0.58 else 1.0
        pixel = _blend_over(pixel, (rr, rg, rb, a_right * outer_fade))

    # Central sculpted "8" (cyan -> azure -> violet-magenta gradient + subtle specular sheen)
    a_eight = _clamp01(0.5 - d_eight / aa)
    if a_eight > 0.001:
        t8 = _clamp01((nx * 0.66 + ny * 0.76 + 0.86) / 1.72)
        if t8 < 0.46:
            er, eg, eb = _lerp_rgb((77.0, 245.0, 255.0), (92.0, 141.0, 255.0), t8 / 0.46)
        else:
            er, eg, eb = _lerp_rgb(
                (92.0, 141.0, 255.0), (238.0, 107.0, 255.0), (t8 - 0.46) / 0.54
            )
        # Subtle glossy top highlight on the ribbon
        sheen = _clamp01(1.0 - abs(d_eight + 0.032) / 0.045) * _clamp01(0.65 - ny * 0.45) * 0.22
        er = _lerp(er, 255.0, sheen)
        eg = _lerp(eg, 255.0, sheen)
        eb = _lerp(eb, 255.0, sheen)
        pixel = _blend_over(pixel, (er, eg, eb, a_eight))

    # Luminous acoustic dots inside upper & lower chambers of the "8"
    d_dot_top = _sdf_circle(nx, ny, 0.0, -0.2890625, 0.046875)
    a_dot_top = _clamp01(0.5 - d_dot_top / aa) * 0.92
    if a_dot_top > 0.001:
        pixel = _blend_over(pixel, (216.0, 251.0, 255.0, a_dot_top))

    d_dot_bot = _sdf_circle(nx, ny, 0.0, 0.25, 0.05859375)
    a_dot_bot = _clamp01(0.5 - d_dot_bot / aa) * 0.92
    if a_dot_bot > 0.001:
        pixel = _blend_over(pixel, (255.0, 217.0, 255.0, a_dot_bot))

    return pixel


def _render_rgba(size: int) -> bytes:
    """Render a supersampled anti-aliased `size x size` RGBA icon buffer."""
    out = bytearray(size * size * 4)
    # 3x3 supersampling for small icons (16..64px) keeps thin edges razor-sharp; 2x2 for 128/256px
    ss = 3 if size <= 64 else 2
    aa = 2.0 / (size * ss)
    inv_n = 1.0 / (ss * ss)
    sub_offsets = tuple((k + 0.5) / ss for k in range(ss))

    idx = 0
    for y in range(size):
        for x in range(size):
            acc_r = 0.0
            acc_g = 0.0
            acc_b = 0.0
            acc_a = 0.0
            for oy in sub_offsets:
                ny = (2.0 * (y + oy) / size) - 1.0
                for ox in sub_offsets:
                    nx = (2.0 * (x + ox) / size) - 1.0
                    r, g, b, a = _sample_icon(nx, ny, aa)
                    acc_r += r * a
                    acc_g += g * a
                    acc_b += b * a
                    acc_a += a
            avg_a = acc_a * inv_n
            if avg_a > 1e-6:
                out[idx] = _clamp_u8(acc_r / acc_a)
                out[idx + 1] = _clamp_u8(acc_g / acc_a)
                out[idx + 2] = _clamp_u8(acc_b / acc_a)
                out[idx + 3] = _clamp_u8(avg_a * 255.0)
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


def generate_icons(output_dir: Path, *, static_dir: Path | None = None) -> tuple[Path, Path]:
    """Generate `audio8_vc.ico`, `audio8_vc.png`, and `audio8_vc.svg` inside `output_dir`.

    When `static_dir` is provided, also syncs `favicon.ico`, `icon.png`, and `icon.svg`
    so the WebUI and installed Python wheels always have matching icon assets.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ico_path = output_dir / "audio8_vc.ico"
    png_path = output_dir / "audio8_vc.png"
    svg_path = output_dir / "audio8_vc.svg"

    entries: list[tuple[int, bytes]] = []
    png_256 = b""
    for size in ICO_SIZES:
        rgba = _render_rgba(size)
        if size == 256:
            png_256 = encode_png(size, rgba)
            entries.append((size, png_256))
        else:
            entries.append((size, _encode_ico_dib(size, rgba)))

    # Write 256x256 PNG and master SVG
    png_path.write_bytes(png_256)
    svg_path.write_text(ICON_SVG.strip() + "\n", encoding="utf-8")

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

    ico_bytes = ico_header + bytes(dir_entries) + bytes(payloads)
    ico_path.write_bytes(ico_bytes)

    if static_dir is not None:
        static_dir.mkdir(parents=True, exist_ok=True)
        (static_dir / "favicon.ico").write_bytes(ico_bytes)
        (static_dir / "icon.png").write_bytes(png_256)
        (static_dir / "icon.svg").write_text(ICON_SVG.strip() + "\n", encoding="utf-8")

    return ico_path, png_path


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    out_dir = script_dir / "assets"
    static_dir = repo_root / "src" / "a8vc" / "server" / "static"
    ico_path, png_path = generate_icons(out_dir, static_dir=static_dir)
    svg_path = out_dir / "audio8_vc.svg"
    print(f"Generated {ico_path} ({ico_path.stat().st_size} bytes)")
    print(f"Generated {png_path} ({png_path.stat().st_size} bytes)")
    print(f"Generated {svg_path} ({svg_path.stat().st_size} bytes)")
    print(f"Synced WebUI icons to {static_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
