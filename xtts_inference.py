#!/usr/bin/env python3
#
import torch
import torchaudio
import argparse
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from typing import Tuple, List

"""
Example usage:
./xtts_inference.py --config run/training/GPT_XTTS_Portuguese-November-07-2023_10+41PM-45891d00/config.json \
--tokenizer run/training/XTTS_v2.0_original_model_files/vocab.json \
--checkpoint run/training/GPT_XTTS_Portuguese-November-07-2023_10+41PM-45891d00/best_model.pth \
--speaker_ref datasets/ptbr_iara/audios/bernardohenz@gmail.com/undefined/1681416644_26554169.wav \
--text "a radiografia apresentou fratura no fêmur esquerdo ponto nova linha" \
--language "pt" --output test.wav
"""


def load_model(
    config_path: str, tokenizer_path: str, checkpoint_path: str, use_deepspeed: bool = False, device: str = "cpu"
) -> Xtts:
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config, checkpoint_path=checkpoint_path, vocab_path=tokenizer_path, use_deepspeed=use_deepspeed
    )
    if device == "cuda":
        model.cuda()
    return model


def compute_speaker_latents(model: Xtts, speaker_reference_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_reference_paths)
    return gpt_cond_latent, speaker_embedding


def perform_inference(
    model: Xtts,
    text: str,
    language: str,
    gpt_cond_latent: torch.Tensor,
    speaker_embedding: torch.Tensor,
    temperature: float = 0.7,
) -> dict:
    out = model.inference(
        text=text,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=temperature,
    )
    return out


def save_output_wav(output_path: str, waveform: torch.Tensor) -> None:
    torchaudio.save(output_path, waveform.unsqueeze(0), 24000)


def main():
    parser = argparse.ArgumentParser(description="XTTS Inference Script")
    parser.add_argument("--config", required=True, help="Path to the XTTS configuration file")
    parser.add_argument("--tokenizer", required=True, help="Path to the tokenizer vocabulary file")
    parser.add_argument("--checkpoint", required=True, help="Path to the XTTS checkpoint file")
    parser.add_argument("--speaker_ref", required=True, help="Path to the speaker reference audio", nargs="+")
    parser.add_argument("--output", required=True, help="Path to save the output WAV file")
    parser.add_argument("--text", required=True, help="Input text for synthesis")
    parser.add_argument("--language", help="Language for synthesis", default="pt")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for model (cpu or cuda)")

    args = parser.parse_args()

    model = load_model(args.config, args.tokenizer, args.checkpoint, device=args.device)
    gpt_cond_latent, speaker_embedding = compute_speaker_latents(model, args.speaker_ref)
    out = perform_inference(model, args.text, args.language, gpt_cond_latent, speaker_embedding)
    save_output_wav(args.output, torch.tensor(out["wav"]))


if __name__ == "__main__":
    main()
