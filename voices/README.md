# Voices Directory

Place custom `.a8voice` single-file voice packs here, or create them using:

```bash
uv run a8vc voice pack --audio /path/to/ref.wav --id my_voice --text "参考音频的逐字文本"
```

## Built-in Open-Source Preset Voices

Audio8-VC bundles **15 curated open-source `.a8voice` presets** (~47 KB total) inside `src/a8vc/voices/builtins/` (covering anime characters, game characters, Chinese multi-style, and top-ranked English voices from OpenBMB/VoxCPM, FunAudioLLM/CosyVoice, hexgrad/Kokoro-82M, kyutai/tts-voices, Qwen3-TTS ITA-Corpus, and Edge0-AI/Audio8_TTS).

- On first launch of `a8vc serve` or the Windows desktop app (`Audio8-VC.exe`), all built-in voices are automatically seeded into your active `voices/` directory.
- From the CLI, you can inspect and install them anytime:

```bash
uv run a8vc voice builtins
uv run a8vc voice install-builtins
```

See `src/a8vc/voices/builtins/ATTRIBUTION.md` for full license and attribution details.

`.a8voice` files placed in this folder are ignored by Git (see `.gitignore`).
