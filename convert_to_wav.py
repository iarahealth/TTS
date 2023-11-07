#!/usr/bin/env python3

import os
import argparse
import pandas as pd

from pydub import AudioSegment
from tqdm import tqdm


def convert_ogg_to_wav(file_path):
    ogg_audio = AudioSegment.from_ogg(file_path)
    ogg_audio = ogg_audio.set_frame_rate(22050)
    wav_file = file_path.replace(".ogg", ".wav")
    ogg_audio.export(wav_file, format="wav")
    return wav_file


def replace_ogg_with_wav(folder_path):
    for root, _, files in os.walk(folder_path):
        for f in tqdm(files):
            if f.endswith(".ogg"):
                ogg_path = os.path.join(root, f)
                convert_ogg_to_wav(ogg_path)
                os.remove(ogg_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert .ogg files to .wav, resampling to 22050 Hz, and replace them in a folder."
    )
    parser.add_argument("folder_path", help="Path to the folder containing .ogg files", type=str)
    parser.add_argument("--meta", help="Path with metadata file", default=None, type=str)
    args = parser.parse_args()

    folder_path = args.folder_path
    replace_ogg_with_wav(folder_path)
    print("Conversion, resampling, and replacement completed.")

    if args.meta:
        df = pd.read_csv(args.meta, sep="|")
        df = df.drop_duplicates(keep="first")
        df["wav_filename"] = df["wav_filename"].str.replace(".ogg", ".wav")
        df.to_csv("meta_clean.tsv", index=False, sep="|")


if __name__ == "__main__":
    main()
