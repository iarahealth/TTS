#!/usr/bin/env python3

import os
import argparse
import pandas as pd

from pydub import AudioSegment
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def convert_ogg_to_wav(file_path):
    ogg_audio = AudioSegment.from_ogg(file_path)
    ogg_audio = ogg_audio.set_frame_rate(22050)
    ogg_audio = ogg_audio.set_channels(1)
    wav_file = file_path.replace(".ogg", ".wav")
    ogg_audio.export(wav_file, format="wav")
    return wav_file


def process_ogg_file(file_path):
    convert_ogg_to_wav(file_path)
    os.remove(file_path)


def replace_ogg_with_wav(folder_path):
    files_to_process = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(".ogg"):
                ogg_path = os.path.join(root, f)
                files_to_process.append(ogg_path)

    num_cores = cpu_count()
    with Pool(processes=num_cores) as pool:
        list(tqdm(pool.imap(process_ogg_file, files_to_process), total=len(files_to_process)))


def main():
    parser = argparse.ArgumentParser(
        description="Convert .ogg files to .wav, resampling to 22050 Hz, and replace them in a folder."
    )
    parser.add_argument("--folder_paths", nargs="+", help="Paths to the folders containing .ogg files", default=[])
    parser.add_argument("--meta", help="Path with metadata file", default=None, type=str)
    args = parser.parse_args()

    folder_paths = args.folder_paths
    for folder_path in folder_paths:
        print(f"Processing folder: {folder_path}")
        replace_ogg_with_wav(folder_path)
    print("Conversion, resampling, and replacement completed.")

    if args.meta:
        df = pd.read_csv(args.meta, sep="|")
        df = df.drop_duplicates(keep="first")
        df["wav_filename"] = df["wav_filename"].str.replace(".ogg", ".wav")
        df.to_csv("meta_clean.tsv", index=False, sep="|")


if __name__ == "__main__":
    main()
