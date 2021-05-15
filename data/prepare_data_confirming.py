import sys
sys.path.append("../")

import torch.nn as nn
import torch
import librosa
import os
import numpy as np
import json

DATASET_PATH = ["../../confirming_dataset/yes", "../../confirming_dataset/no", "../../confirming_dataset/unknown"]
SAVED_FILE = "confirming_data/data.json"

def preprocess_dataset(dataset_path, saved_file_path, n_mels=40, n_fft=512, hop_length=384, fmax=8000):
    saved_data = []
    dataset_number = 0

    for data_set in dataset_path:

        # dictionary where we'll store mapping, labels, MFCCs and filenames
        data = {
            "mel_spectrogram": [],
            "labels": [],
        }

        # loop through all sub-dirs
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_set)):

            label = dirpath.split("/")[-1]

            print("\nProcessing: '{}'".format(label))

            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)

                # extract MFCCs (#features, #time binz)
                mel_spectrogram = librosa.feature.melspectrogram(signal, sample_rate, n_mels=n_mels, n_fft=n_fft,
                                                hop_length=hop_length, fmax=fmax)

                mel_spectrogram = librosa.power_to_db(mel_spectrogram)

                # store data for analysed track
                data["mel_spectrogram"].append(mel_spectrogram.T.tolist())
                data["labels"].append(dataset_number)
                print("{}: {}".format(file_path, dataset_number))

            saved_data.append(data)
            dataset_number += 1

    with open(saved_file_path, 'w') as f:
        json.dump(saved_data, f, indent=4)


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, SAVED_FILE)