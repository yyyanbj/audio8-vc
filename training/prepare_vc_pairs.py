#!/usr/bin/env python3
"""Prepare multi-speaker `[10, T]` codec token datasets for Track 2 S2S VC fine-tuning.

Input JSONL format (one utterance per line):
    {"id": "spk01_0001", "speaker": "spk01", "audio": "/abs/path/utt1.wav", "text": "..."}

Outputs:
- Precomputed target `[10, T]` uint16 `.npy` files under `--out-dir/codes/`
- Paired S2S manifest `vc_pairs_prepared.jsonl` linking `(ref_codes, target_codes, speaker)`
  so same-speaker reference/target pairs can be trained with online timbre perturbation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from a8vc.audio.io import load_audio
from a8vc.voices.enroll import VoiceEnroller


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Track 2 S2S VC codec pairs")
    parser.add_argument("--input-jsonl", required=True, help="Raw multi-speaker utterance JSONL")
    parser.add_argument("--model-dir", default="models/audio8-tts-0.1b-onnx-int8")
    parser.add_argument("--out-dir", default="training/data")
    parser.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    in_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    codes_dir = out_dir / "codes"
    codes_dir.mkdir(parents=True, exist_ok=True)

    enroller = VoiceEnroller(args.model_dir, provider=args.provider, threads=args.threads)
    by_speaker: dict[str, list[dict[str, object]]] = defaultdict(list)

    with in_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            utt_id = str(item["id"])
            speaker = str(item["speaker"])
            audio_path = Path(item["audio"])
            audio, sr = load_audio(audio_path)
            codes, _ = enroller.encode_waveform(audio, sr)
            rel_code_path = Path("codes") / f"{utt_id}.npy"
            np.save(str(out_dir / rel_code_path), codes.astype(np.uint16), allow_pickle=False)
            by_speaker[speaker].append(
                {
                    "id": utt_id,
                    "speaker": speaker,
                    "codes_path": str(rel_code_path),
                    "num_frames": int(codes.shape[1]),
                    "text": str(item.get("text", "")),
                }
            )
            if line_no % 50 == 0:
                print(f"Encoded {line_no} utterances...")

    manifest_path = out_dir / "vc_pairs_prepared.jsonl"
    pair_count = 0
    with manifest_path.open("w", encoding="utf-8") as out_fh:
        for speaker, utts in by_speaker.items():
            if len(utts) < 2:
                continue
            for idx, target in enumerate(utts):
                ref = utts[(idx + 1) % len(utts)]
                record = {
                    "id": f"{speaker}_pair_{idx:05d}",
                    "speaker": speaker,
                    "ref_codes_path": ref["codes_path"],
                    "ref_text": ref["text"],
                    "target_codes_path": target["codes_path"],
                    "target_text": target["text"],
                    "num_frames": target["num_frames"],
                }
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                pair_count += 1

    print(f"Wrote {pair_count} same-speaker VC pairs -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
