#!/usr/bin/env python3
import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"

# List available Coqui TTS models
# print(TTS().list_models())

# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1.1").to(device)
# tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to(device)
# tts = TTS("tts_models/por/fairseq/vits").to(device)
tts = TTS("tts_models/pt/cv/vits").to(device)

tts.tts_with_vc_to_file(
    text="A radiografia apresentou algumas lesões no fêmur esquerdo ponto parágrafo",
    speaker_wav="test_audios/1693678335_24253176-processed.wav",
    file_path="test_audios/output.wav",  # Do not save OGG or there will be some audio crackling!
)

# Use below if using xtts_v1.1 or your_tts.
"""
tts.tts_to_file(
    text="A radiografia apresentou algumas lesões no fêmur esquerdo ponto parágrafo",
    speaker_wav="test_audios/1693678335_24253176-processed.wav"",
    language="pt", # or "pt-br" for your_tts
    file_path="test_audios/output.wav",
)
"""
