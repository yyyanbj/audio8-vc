# Built-in Voice Packs Attribution & Licenses

All built-in `.a8voice` voice packs bundled in `a8vc/voices/builtins/` contain only compact
discrete Audio8-0.1B codec token sequences (typically 3–6 KB each) derived from permissively
licensed open-source reference audio clips and official Audio8 pre-encoded codes.

| Voice ID | Name | Category | Lang | Source Project | License | Attribution |
|---|---|---|---|---|---|---|
| `voxcpm_tsundere` | 傲娇少女 · 冰淇淋 | 二次元 | `zh` | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) | `Apache-2.0` | OpenBMB/VoxCPM (examples/reference_speaker.wav), Apache-2.0 |
| `zh_lively_girl` | 娇俏少女 · 高音元气 | 二次元 | `zh` | [Edge0-AI/Audio8_TTS](https://github.com/Edge0-AI/Audio8_TTS) | `Apache-2.0` | Edge0-AI/Audio8_TTS (docs/0.1B/audio/reference/zh_04.mp3), Apache-2.0 |
| `anime_loli_jp` | 日系软萌萝莉 (JA) | 二次元 | `ja` | [Qwen3-TTS / Akjava ITA-Corpus Emotion](https://huggingface.co/datasets/Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion) | `MIT` | Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion (qwen100_24khz/002/015.wav, synthesized via Qwen3-TTS, script: public-domain ITA-Corpus), MIT |
| `anime_genki_jp` | 日系元气少女 (JA) | 二次元 | `ja` | [Qwen3-TTS / Akjava ITA-Corpus Emotion](https://huggingface.co/datasets/Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion) | `MIT` | Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion (qwen100_24khz/006/015.wav, synthesized via Qwen3-TTS, script: public-domain ITA-Corpus), MIT |
| `anime_oneesan_jp` | 日系酷飒御姐 (JA) | 二次元 | `ja` | [Qwen3-TTS / Akjava ITA-Corpus Emotion](https://huggingface.co/datasets/Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion) | `MIT` | Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion (qwen100_24khz/010/027.wav, synthesized via Qwen3-TTS, script: public-domain ITA-Corpus), MIT |
| `anime_shonen_jp` | 日系清朗少年 (JA) | 二次元 | `ja` | [Qwen3-TTS / Akjava ITA-Corpus Emotion](https://huggingface.co/datasets/Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion) | `MIT` | Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion (qwen100_24khz/015/015.wav, synthesized via Qwen3-TTS, script: public-domain ITA-Corpus), MIT |
| `game_announcer_en` | 竞技游戏播报 · Announcer | 游戏角色 | `en` | [kyutai/tts-voices (Alba MacKenna)](https://huggingface.co/kyutai/tts-voices) | `CC-BY-4.0` | Alba MacKenna via kyutai/tts-voices (alba-mackenna/announcer.wav), CC-BY-4.0 |
| `game_narrator_en` | 游戏治愈旁白 · Kinder World | 游戏角色 | `en` | [kyutai/tts-voices (Alba MacKenna)](https://huggingface.co/kyutai/tts-voices) | `CC-BY-4.0` | Alba MacKenna via kyutai/tts-voices (alba-mackenna/a-moment-by.wav), CC-BY-4.0 |
| `cosyvoice_sweet` | 甜美少女 · CosyVoice | 中文精选 | `zh` | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | `Apache-2.0` | FunAudioLLM/CosyVoice (asset/zero_shot_prompt.wav), Apache-2.0 |
| `zh_gentle_narrator` | 温柔治愈 · 故事女声 | 中文精选 | `zh` | [Edge0-AI/Audio8_TTS](https://github.com/Edge0-AI/Audio8_TTS) | `Apache-2.0` | Edge0-AI/Audio8_TTS (docs/0.1B/audio/reference/zh_06.mp3), Apache-2.0 |
| `zh_whisper_female` | 温柔轻语 · 晚安女声 | 中文精选 | `zh` | [Edge0-AI/Audio8_TTS](https://github.com/Edge0-AI/Audio8_TTS) | `Apache-2.0` | Edge0-AI/Audio8_TTS (docs/0.1B/audio/reference/zh_14.mp3), Apache-2.0 |
| `zh_news_male` | 沉稳男播 · 气象播报 | 中文精选 | `zh` | [Edge0-AI/Audio8_TTS](https://github.com/Edge0-AI/Audio8_TTS) | `Apache-2.0` | Edge0-AI/Audio8_TTS (docs/0.1B/audio/reference/zh_02.mp3), Apache-2.0 |
| `edge0_default` | 官方默认 · 元气火箭 | 中文精选 | `zh` | [Edge0/audio8-TTS-0.1B-ONNX-INT8](https://huggingface.co/Edge0/audio8-TTS-0.1B-ONNX-INT8) | `Audio8-Community-1.0` | Edge0/audio8-TTS-0.1B-ONNX-INT8 (reference_codes.npy), Audio8 Community License v1.0 |
| `kokoro_heart_en` | Kokoro 榜首女声 · Heart | 英文精选 | `en` | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | `Apache-2.0` | hexgrad/Kokoro-82M (samples/af_heart_0.wav), Apache-2.0 |
| `pocket_narrator_male` | 磁性英伦男声 · 叙事 | 英文精选 | `en` | [kyutai/tts-voices (Voice-Zero / Pocket TTS)](https://huggingface.co/kyutai/tts-voices) | `CC0-1.0` | Peter Yearsley (LibriVox) via OwenTyme/voice-zero & kyutai/tts-voices (voice-zero/peter_yearsley.wav), CC0-1.0 |
