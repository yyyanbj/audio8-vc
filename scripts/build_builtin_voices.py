#!/usr/bin/env python3
"""Build and verify the curated built-in `.a8voice` preset library from open-source voices.

This script downloads permissively licensed voice reference clips from top-ranked open-source
TTS projects (OpenBMB/VoxCPM, FunAudioLLM/CosyVoice, hexgrad/Kokoro-82M, kyutai/tts-voices,
Qwen3-TTS ITA-Corpus Emotion, and Edge0-AI/Audio8_TTS), encodes each clip using Audio8-0.1B's
`codec_encoder_fp16.onnx` (or imports Edge0's official pre-encoded `reference_codes.npy`),
verifies SHA-256 and codebook bounds, and writes:

- `src/a8vc/voices/builtins/<voice_id>.a8voice`
- `src/a8vc/voices/builtins/catalog.json`
- `src/a8vc/voices/builtins/ATTRIBUTION.md`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from a8vc.audio.io import load_audio, pad_to_multiple, resample_audio
from a8vc.runtime.providers import create_onnx_session
from a8vc.voices.enroll import CODEC_FRAME_ALIGN
from a8vc.voices.format import (
    A8VOICE_EXT,
    build_voice_pack,
    save_voice_pack,
    validate_codes_array,
    verify_voice_pack,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "src" / "a8vc" / "voices" / "builtins"


@dataclass(frozen=True)
class BuiltinVoiceSpec:
    voice_id: str
    name: str
    category: str  # "anime" | "game" | "zh" | "en"
    category_label: str
    language: str
    description: str
    reference_text: str
    tags: tuple[str, ...]
    source_project: str
    source_rank_note: str
    source_url: str
    license: str
    attribution: str
    audio_url: str | None = None  # None means use official pre-encoded reference_codes.npy
    cache_filename: str = ""
    trim_start_sec: float | None = None
    trim_end_sec: float | None = None
    target_rms: float | None = None


CURATED_SPECS: tuple[BuiltinVoiceSpec, ...] = (
    # --- Anime / 二次元角色音色 ---
    BuiltinVoiceSpec(
        voice_id="voxcpm_tsundere",
        name="傲娇少女 · 冰淇淋",
        category="anime",
        category_label="二次元",
        language="zh",
        description="活泼灵动的高音中文傲娇二次元少女声线，情绪张力强，中文变声清晰自然。",
        reference_text=(
            "喂喂喂，说你呢！要是再不把桌子上的冰淇淋吃掉，我可就要没收了哦！"
            "别以为我不知道你昨天晚上又熬夜打游戏了，下次再这样，我真的要生气啦，哼！"
        ),
        tags=("builtin", "anime", "tsundere", "female", "zh", "voxcpm"),
        source_project="OpenBMB/VoxCPM",
        source_rank_note="GitHub 43k+ Stars (#1 开源 TTS)",
        source_url="https://github.com/OpenBMB/VoxCPM",
        license="Apache-2.0",
        attribution="OpenBMB/VoxCPM (examples/reference_speaker.wav), Apache-2.0",
        audio_url="https://raw.githubusercontent.com/OpenBMB/VoxCPM/main/examples/reference_speaker.wav",
        cache_filename="voxcpm_reference_speaker.wav",
    ),
    BuiltinVoiceSpec(
        voice_id="zh_lively_girl",
        name="娇俏少女 · 高音元气",
        category="anime",
        category_label="二次元",
        language="zh",
        description="清脆高音且富有情绪张力的中文年轻少女声线（~329Hz），适合中文二次元/直播互动。",
        reference_text="这份报告明明昨天就该交，你现在才告诉我数据还没有核对完，怎么能这样？",
        tags=("builtin", "anime", "energetic", "female", "zh", "edge0"),
        source_project="Edge0-AI/Audio8_TTS",
        source_rank_note="Audio8 官方 0.1B 示例参考音色",
        source_url="https://github.com/Edge0-AI/Audio8_TTS",
        license="Apache-2.0",
        attribution="Edge0-AI/Audio8_TTS (docs/0.1B/audio/reference/zh_04.mp3), Apache-2.0",
        audio_url="https://raw.githubusercontent.com/Edge0-AI/Audio8_TTS/master/docs/0.1B/audio/reference/zh_04.mp3",
        cache_filename="edge0_zh_04.mp3",
    ),
    BuiltinVoiceSpec(
        voice_id="anime_loli_jp",
        name="日系软萌萝莉 (JA)",
        category="anime",
        category_label="二次元",
        language="ja",
        description="偏高稚嫩的日系软萌萝莉声线（说日语效果最佳，说中文带日系口音）。",
        reference_text="どーすんの、このお店。完ッ全に閑古鳥が鳴いちゃってるじゃない。",
        tags=("builtin", "anime", "loli", "cute", "female", "ja", "qwen3-tts"),
        source_project="Qwen3-TTS / Akjava ITA-Corpus Emotion",
        source_rank_note="Qwen3-TTS 14.6k+ Stars 衍生合成声库",
        source_url="https://huggingface.co/datasets/Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion",
        license="MIT",
        attribution=(
            "Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion "
            "(qwen100_24khz/002/015.wav, synthesized via Qwen3-TTS, script: public-domain ITA-Corpus), MIT"
        ),
        audio_url=(
            "https://huggingface.co/datasets/Akjava/"
            "QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion/"
            "resolve/main/qwen100_24khz/002/015.wav"
        ),
        cache_filename="akjava_002_015.wav",
        target_rms=0.08,
    ),
    BuiltinVoiceSpec(
        voice_id="anime_genki_jp",
        name="日系元气少女 (JA)",
        category="anime",
        category_label="二次元",
        language="ja",
        description="明亮有张力的日系元气女主角声线（说日语效果最佳，说中文带日系口音）。",
        reference_text="どーすんの、このお店。完ッ全に閑古鳥が鳴いちゃってるじゃない。",
        tags=("builtin", "anime", "genki", "energetic", "female", "ja", "qwen3-tts"),
        source_project="Qwen3-TTS / Akjava ITA-Corpus Emotion",
        source_rank_note="Qwen3-TTS 14.6k+ Stars 衍生合成声库",
        source_url="https://huggingface.co/datasets/Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion",
        license="MIT",
        attribution=(
            "Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion "
            "(qwen100_24khz/006/015.wav, synthesized via Qwen3-TTS, script: public-domain ITA-Corpus), MIT"
        ),
        audio_url=(
            "https://huggingface.co/datasets/Akjava/"
            "QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion/"
            "resolve/main/qwen100_24khz/006/015.wav"
        ),
        cache_filename="akjava_006_015.wav",
        target_rms=0.08,
    ),
    BuiltinVoiceSpec(
        voice_id="anime_oneesan_jp",
        name="日系酷飒御姐 (JA)",
        category="anime",
        category_label="二次元",
        language="ja",
        description="偏低沉稳、充满气场的日系酷飒御姐声线（说日语效果最佳，说中文带日系口音）。",
        reference_text="彼女はハンドバッグを開けて家の鍵を探してみたが、見つからなかった。",
        tags=("builtin", "anime", "oneesan", "mature", "female", "ja", "qwen3-tts"),
        source_project="Qwen3-TTS / Akjava ITA-Corpus Emotion",
        source_rank_note="Qwen3-TTS 14.6k+ Stars 衍生合成声库",
        source_url="https://huggingface.co/datasets/Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion",
        license="MIT",
        attribution=(
            "Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion "
            "(qwen100_24khz/010/027.wav, synthesized via Qwen3-TTS, script: public-domain ITA-Corpus), MIT"
        ),
        audio_url=(
            "https://huggingface.co/datasets/Akjava/"
            "QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion/"
            "resolve/main/qwen100_24khz/010/027.wav"
        ),
        cache_filename="akjava_010_027.wav",
        target_rms=0.08,
    ),
    BuiltinVoiceSpec(
        voice_id="anime_shonen_jp",
        name="日系清朗少年 (JA)",
        category="anime",
        category_label="二次元",
        language="ja",
        description="轻快清爽的中性日漫少年感声线（说日语效果最佳，说中文带日系口音）。",
        reference_text="どーすんの、このお店。完ッ全に閑古鳥が鳴いちゃってるじゃない。",
        tags=("builtin", "anime", "shonen", "androgynous", "ja", "qwen3-tts"),
        source_project="Qwen3-TTS / Akjava ITA-Corpus Emotion",
        source_rank_note="Qwen3-TTS 14.6k+ Stars 衍生合成声库",
        source_url="https://huggingface.co/datasets/Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion",
        license="MIT",
        attribution=(
            "Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion "
            "(qwen100_24khz/015/015.wav, synthesized via Qwen3-TTS, script: public-domain ITA-Corpus), MIT"
        ),
        audio_url=(
            "https://huggingface.co/datasets/Akjava/"
            "QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion/"
            "resolve/main/qwen100_24khz/015/015.wav"
        ),
        cache_filename="akjava_015_015.wav",
        target_rms=0.08,
    ),
    # --- Game Characters / 游戏角色音色 ---
    BuiltinVoiceSpec(
        voice_id="game_announcer_en",
        name="竞技游戏播报 · Announcer",
        category="game",
        category_label="游戏角色",
        language="en",
        description="竞技格斗/FPS 风格游戏系统播报员（Alba MacKenna 专业配音），声线饱满有冲击力。",
        reference_text=(
            "No contest! Online! Player One! Player Two! Player Three! Player Four! Player Five!"
        ),
        tags=("builtin", "game", "announcer", "esports", "en", "kyutai"),
        source_project="kyutai/tts-voices (Alba MacKenna)",
        source_rank_note="HF #1 热门开源音色库 (215+ Likes)",
        source_url="https://huggingface.co/kyutai/tts-voices",
        license="CC-BY-4.0",
        attribution="Alba MacKenna via kyutai/tts-voices (alba-mackenna/announcer.wav), CC-BY-4.0",
        audio_url="https://huggingface.co/kyutai/tts-voices/resolve/main/alba-mackenna/announcer.wav",
        cache_filename="game_announcer.wav",
        trim_start_sec=0.95,
    ),
    BuiltinVoiceSpec(
        voice_id="game_narrator_en",
        name="游戏治愈旁白 · Kinder World",
        category="game",
        category_label="游戏角色",
        language="en",
        description="温暖松弛的独立游戏剧情旁白（Alba MacKenna 配音），适合讲故事与沉浸式陪伴。",
        reference_text=(
            "The scent feels familiar to you, strangely nostalgic, "
            "as it brings you back to your favourite childhood meal."
        ),
        tags=("builtin", "game", "narrator", "cozy", "en", "kyutai"),
        source_project="kyutai/tts-voices (Alba MacKenna)",
        source_rank_note="HF #1 热门开源音色库 (215+ Likes)",
        source_url="https://huggingface.co/kyutai/tts-voices",
        license="CC-BY-4.0",
        attribution="Alba MacKenna via kyutai/tts-voices (alba-mackenna/a-moment-by.wav), CC-BY-4.0",
        audio_url="https://huggingface.co/kyutai/tts-voices/resolve/main/alba-mackenna/a-moment-by.wav",
        cache_filename="game_a_moment_by.wav",
    ),
    # --- Chinese Curated / 中文多风格音色 ---
    BuiltinVoiceSpec(
        voice_id="cosyvoice_sweet",
        name="甜美少女 · CosyVoice",
        category="zh",
        category_label="中文精选",
        language="zh",
        description="清甜俏皮的中文年轻女声，吐字清晰、亲和力高，中文日常对话自然流畅。",
        reference_text="希望你以后能够做的比我还好呦。",
        tags=("builtin", "zh", "sweet", "female", "cosyvoice"),
        source_project="FunAudioLLM/CosyVoice",
        source_rank_note="GitHub 24.5k+ Stars",
        source_url="https://github.com/FunAudioLLM/CosyVoice",
        license="Apache-2.0",
        attribution="FunAudioLLM/CosyVoice (asset/zero_shot_prompt.wav), Apache-2.0",
        audio_url="https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav",
        cache_filename="cosyvoice_sweet.wav",
    ),
    BuiltinVoiceSpec(
        voice_id="zh_gentle_narrator",
        name="温柔治愈 · 故事女声",
        category="zh",
        category_label="中文精选",
        language="zh",
        description="温润清澈的治愈系中文女声，适合有声故事、情感陪伴与柔和对话。",
        reference_text="清晨的薄雾沿着山谷慢慢散开，第一缕阳光落在安静的湖面上。",
        tags=("builtin", "zh", "gentle", "narrator", "female", "edge0"),
        source_project="Edge0-AI/Audio8_TTS",
        source_rank_note="Audio8 官方 0.1B 示例参考音色",
        source_url="https://github.com/Edge0-AI/Audio8_TTS",
        license="Apache-2.0",
        attribution="Edge0-AI/Audio8_TTS (docs/0.1B/audio/reference/zh_06.mp3), Apache-2.0",
        audio_url="https://raw.githubusercontent.com/Edge0-AI/Audio8_TTS/master/docs/0.1B/audio/reference/zh_06.mp3",
        cache_filename="edge0_zh_06.mp3",
    ),
    BuiltinVoiceSpec(
        voice_id="zh_whisper_female",
        name="温柔轻语 · 晚安女声",
        category="zh",
        category_label="中文精选",
        language="zh",
        description="轻柔贴耳的治愈系低声女声，适合睡前聊天、电台或轻柔人设。",
        reference_text="灯已经关了，大家都睡着了，我们轻一点说话，别把他们吵醒。",
        tags=("builtin", "zh", "gentle", "whisper", "female", "edge0"),
        source_project="Edge0-AI/Audio8_TTS",
        source_rank_note="Audio8 官方 0.1B 示例参考音色",
        source_url="https://github.com/Edge0-AI/Audio8_TTS",
        license="Apache-2.0",
        attribution="Edge0-AI/Audio8_TTS (docs/0.1B/audio/reference/zh_14.mp3), Apache-2.0",
        audio_url="https://raw.githubusercontent.com/Edge0-AI/Audio8_TTS/master/docs/0.1B/audio/reference/zh_14.mp3",
        cache_filename="edge0_zh_14.mp3",
    ),
    BuiltinVoiceSpec(
        voice_id="zh_news_male",
        name="沉稳男播 · 气象播报",
        category="zh",
        category_label="中文精选",
        language="zh",
        description="稳重清晰的标准普通话男声，播音腔饱满，适合配音解说与正式语音。",
        reference_text="根据最新气象资料，本周末北方多地将迎来降温，出行前请及时查看天气预报。",
        tags=("builtin", "zh", "male", "broadcaster", "edge0"),
        source_project="Edge0-AI/Audio8_TTS",
        source_rank_note="Audio8 官方 0.1B 示例参考音色",
        source_url="https://github.com/Edge0-AI/Audio8_TTS",
        license="Apache-2.0",
        attribution="Edge0-AI/Audio8_TTS (docs/0.1B/audio/reference/zh_02.mp3), Apache-2.0",
        audio_url="https://raw.githubusercontent.com/Edge0-AI/Audio8_TTS/master/docs/0.1B/audio/reference/zh_02.mp3",
        cache_filename="edge0_zh_02.mp3",
    ),
    BuiltinVoiceSpec(
        voice_id="edge0_default",
        name="官方默认 · 元气火箭",
        category="zh",
        category_label="中文精选",
        language="zh",
        description="Audio8-TTS-0.1B 官方内置预编码基准音色，稳定性高，适合快速试麦基准测试。",
        reference_text="至今为止，元气火箭总共发行了两张专辑。",
        tags=("builtin", "zh", "official", "default", "edge0"),
        source_project="Edge0/audio8-TTS-0.1B-ONNX-INT8",
        source_rank_note="Audio8-0.1B 官方默认预编码音色",
        source_url="https://huggingface.co/Edge0/audio8-TTS-0.1B-ONNX-INT8",
        license="Audio8-Community-1.0",
        attribution="Edge0/audio8-TTS-0.1B-ONNX-INT8 (reference_codes.npy), Audio8 Community License v1.0",
        audio_url=None,
    ),
    # --- English Top-Ranked / 英文热门开源音色 ---
    BuiltinVoiceSpec(
        voice_id="kokoro_heart_en",
        name="Kokoro 榜首女声 · Heart",
        category="en",
        category_label="英文精选",
        language="en",
        description="Hugging Face 下载量第一的 Kokoro-82M 旗舰 A 级女声 (af_heart)，自然清晰且表现力出众。",
        reference_text="The sky above the port was the color of television, tuned to a dead channel.",
        tags=("builtin", "en", "female", "top-ranked", "kokoro"),
        source_project="hexgrad/Kokoro-82M",
        source_rank_note="HF #1 热门开源 TTS (7.5k+ Likes, 13.7M+ DL)",
        source_url="https://huggingface.co/hexgrad/Kokoro-82M",
        license="Apache-2.0",
        attribution="hexgrad/Kokoro-82M (samples/af_heart_0.wav), Apache-2.0",
        audio_url="https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/samples/af_heart_0.wav",
        cache_filename="kokoro_heart.wav",
    ),
    BuiltinVoiceSpec(
        voice_id="pocket_narrator_male",
        name="磁性英伦男声 · 叙事",
        category="en",
        category_label="英文精选",
        language="en",
        description="低沉磁性的经典英伦叙事男声（Pocket TTS 默认音色，LibriVox 公有领域 CC0）。",
        reference_text=(
            "The North Wind and the Sun were disputing which was the stronger, "
            "when a traveler came along wrapped in a warm cloak."
        ),
        tags=("builtin", "en", "male", "narrator", "deep", "kyutai"),
        source_project="kyutai/tts-voices (Voice-Zero / Pocket TTS)",
        source_rank_note="HF #1 开源音色库 · Pocket TTS 默认男声",
        source_url="https://huggingface.co/kyutai/tts-voices",
        license="CC0-1.0",
        attribution=(
            "Peter Yearsley (LibriVox) via OwenTyme/voice-zero & kyutai/tts-voices "
            "(voice-zero/peter_yearsley.wav), CC0-1.0"
        ),
        audio_url="https://huggingface.co/kyutai/tts-voices/resolve/main/voice-zero/peter_yearsley.wav",
        cache_filename="vz_peter_yearsley.wav",
    ),
)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "Audio8-VC-BuiltinBuilder/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        dest.write_bytes(resp.read())
    return dest


def _encode_audio(
    session: Any,
    audio: np.ndarray,
    orig_sr: int,
    *,
    target_sr: int,
    num_codebooks: int,
    codebook_size: int,
    hop_length: int,
) -> np.ndarray:
    resampled = resample_audio(audio, orig_sr, target_sr)
    padded = pad_to_multiple(resampled, multiple=hop_length)
    tensor = np.ascontiguousarray(padded.reshape(1, 1, -1))
    inp = session.get_inputs()[0]
    feed = tensor.astype(np.float16, copy=False) if inp.type == "tensor(float16)" else tensor.astype(np.float32, copy=False)
    outputs = session.run(None, {inp.name: feed})
    codes = np.asarray(outputs[0])
    if codes.ndim == 3:
        codes = codes[0]
    return validate_codes_array(codes, num_codebooks=num_codebooks, codebook_size=codebook_size)


def _write_attribution(out_dir: Path, entries: list[dict[str, Any]]) -> None:
    lines = [
        "# Built-in Voice Packs Attribution & Licenses",
        "",
        "All built-in `.a8voice` voice packs bundled in `a8vc/voices/builtins/` contain only compact",
        "discrete Audio8-0.1B codec token sequences (typically 3–6 KB each) derived from permissively",
        "licensed open-source reference audio clips and official Audio8 pre-encoded codes.",
        "",
        "| Voice ID | Name | Category | Lang | Source Project | License | Attribution |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        lines.append(
            f"| `{e['voice_id']}` | {e['name']} | {e['category_label']} | `{e['language']}` "
            f"| [{e['source_project']}]({e['source_url']}) | `{e['license']}` | {e['attribution']} |"
        )
    lines.append("")
    (out_dir / "ATTRIBUTION.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build curated built-in .a8voice packs.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/tmp/a8vc_build_cache/audio8"),
        help="Directory containing runtime_manifest.json, reference_codes.npy, and registration/codec_encoder_fp16.onnx",
    )
    parser.add_argument(
        "--raw-cache-dir",
        type=Path,
        default=Path("/tmp/a8vc_build_cache/raw_voices"),
        help="Temporary download cache directory for upstream reference audio clips",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for built-in .a8voice packs and catalog.json",
    )
    parser.add_argument("--threads", type=int, default=4, help="ONNX Runtime CPU threads")
    args = parser.parse_args()

    model_dir: Path = args.model_dir
    encoder_path = model_dir / "registration" / "codec_encoder_fp16.onnx"
    ref_codes_path = model_dir / "reference_codes.npy"
    runtime_manifest_path = model_dir / "runtime_manifest.json"
    reg_manifest_path = model_dir / "registration" / "registration_manifest.json"

    for req_path in (encoder_path, ref_codes_path, runtime_manifest_path):
        if not req_path.is_file():
            raise SystemExit(f"Missing required Audio8 file: {req_path}")

    meta: dict[str, Any] = {
        "sample_rate": 44100,
        "num_codebooks": 10,
        "codebook_size": 4096,
        "codec_hop_length": CODEC_FRAME_ALIGN,
        "model_fingerprint": "",
    }
    for mp in (runtime_manifest_path, reg_manifest_path):
        if mp.is_file():
            meta.update(json.loads(mp.read_text(encoding="utf-8")))

    target_sr = int(meta["sample_rate"])
    num_codebooks = int(meta["num_codebooks"])
    codebook_size = int(meta["codebook_size"])
    hop_length = int(meta["codec_hop_length"])
    fingerprint = str(meta.get("model_fingerprint", ""))

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading codec encoder: {encoder_path}")
    session, active_ep, _ = create_onnx_session(encoder_path, provider="cpu", threads=args.threads)
    print(f"Codec encoder ready (provider={active_ep}, fingerprint={fingerprint})")

    catalog_entries: list[dict[str, Any]] = []
    total_bytes = 0

    for spec in CURATED_SPECS:
        if spec.audio_url is None:
            codes = validate_codes_array(
                np.load(str(ref_codes_path), allow_pickle=False),
                num_codebooks=num_codebooks,
                codebook_size=codebook_size,
            )
            source_name = "reference_codes.npy"
            source_sr = target_sr
            source_sha256 = hashlib.sha256(ref_codes_path.read_bytes()).hexdigest()
        else:
            raw_path = _download(spec.audio_url, args.raw_cache_dir / spec.cache_filename)
            source_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            audio, orig_sr = load_audio(raw_path)
            if spec.trim_start_sec is not None or spec.trim_end_sec is not None:
                start_i = int((spec.trim_start_sec or 0.0) * orig_sr)
                end_i = int(spec.trim_end_sec * orig_sr) if spec.trim_end_sec is not None else len(audio)
                audio = audio[start_i:end_i]
            if spec.target_rms is not None and audio.size > 0:
                rms = float(np.sqrt(np.mean(audio * audio)))
                if rms > 1e-6:
                    scale = spec.target_rms / rms
                    peak = float(np.max(np.abs(audio)))
                    if peak * scale > 0.95:
                        scale = 0.95 / peak
                    audio = (audio * scale).astype(np.float32, copy=False)
            codes = _encode_audio(
                session,
                audio,
                orig_sr,
                target_sr=target_sr,
                num_codebooks=num_codebooks,
                codebook_size=codebook_size,
                hop_length=hop_length,
            )
            source_name = raw_path.name
            source_sr = orig_sr

        pack = build_voice_pack(
            voice_id=spec.voice_id,
            codes=codes,
            reference_text=spec.reference_text,
            name=spec.name,
            description=spec.description,
            language=spec.language,
            tags=list(spec.tags),
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            sample_rate=target_sr,
            codec_hop_length=hop_length,
            source_audio_name=source_name,
            source_sample_rate=source_sr,
            source_sha256=source_sha256,
            model_fingerprint=fingerprint,
        )
        out_path = save_voice_pack(pack, out_dir / f"{spec.voice_id}{A8VOICE_EXT}")
        if os.name != "nt":
            os.chmod(out_path, 0o644)
        report = verify_voice_pack(out_path)
        assert report["valid"] is True
        size_bytes = int(report["size_bytes"])
        total_bytes += size_bytes

        catalog_entries.append(
            {
                "voice_id": spec.voice_id,
                "filename": out_path.name,
                "name": spec.name,
                "category": spec.category,
                "category_label": spec.category_label,
                "language": spec.language,
                "description": spec.description,
                "tags": list(spec.tags),
                "duration_sec": pack.manifest.duration_sec,
                "num_frames": pack.manifest.num_frames,
                "size_bytes": size_bytes,
                "codes_sha256": pack.manifest.codes_sha256,
                "model_fingerprint": fingerprint,
                "source_project": spec.source_project,
                "source_rank_note": spec.source_rank_note,
                "source_url": spec.source_url,
                "license": spec.license,
                "attribution": spec.attribution,
            }
        )
        print(
            f"  [OK] {spec.voice_id:<22} {spec.category_label:<6} lang={spec.language} "
            f"dur={pack.manifest.duration_sec:5.2f}s frames={pack.manifest.num_frames:3d} "
            f"size={size_bytes / 1024:.1f} KB"
        )

    catalog = {
        "schema_version": 1,
        "model_fingerprint": fingerprint,
        "voices": catalog_entries,
    }
    (out_dir / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_attribution(out_dir, catalog_entries)

    print(
        f"\nBuilt {len(catalog_entries)} built-in .a8voice packs "
        f"(total {total_bytes / 1024:.1f} KB) in {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
