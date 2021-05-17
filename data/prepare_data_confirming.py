import sys
sys.path.append("../")

import torch.nn as nn
import torch
import librosa
import os
import numpy as np
import json
import torchaudio
import augment
from scipy.io import wavfile

DATASET_PATH = ["../../confirming_dataset/co", "../../confirming_dataset/khong"]
SAVED_FILE = "confirming_data/data.json"

def preprocess_dataset(dataset_path, saved_file_path):
    saved_data = []
    dataset_number = 0

    # mel spectrogram
    kwargs = {
        'n_fft': 512,
        'n_mels': 40,
    }
    wav_to_spec = torchaudio.transforms.MelSpectrogram(**kwargs)

    # spectrogram augmentation
    kwargs = {
        'rect_freq': 7,
        'rect_masks': 10,
        'rect_time': 40,
    }
    spec_augment = augment.SpectrogramAugmentation(**kwargs)

    for data_set in dataset_path:

        # dictionary where we'll store mapping, labels, MFCCs and filenames
        data_temporary = {
            "mel_spectrogram": [],
            "labels": []
        }

        # loop through all sub-dirs
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_set)):

            label = dirpath.split("/")[-1]

            print("\nProcessing: '{}'".format(label))

            for f in filenames:
                file_path = os.path.join(dirpath, f)

                fs, data = wavfile.read(file_path)
                data = torch.Tensor(data.copy())
                data = data / data.abs().max()

                x = wav_to_spec(data.clone())

                for i in range(100):
                    mel_spectrogram = spec_augment(x.clone().unsqueeze(0)).squeeze(0).permute(0, 1).tolist()

                    # store data for analysed track
                    data_temporary["mel_spectrogram"].append(mel_spectrogram)
                    data_temporary["labels"].append(dataset_number)
                    print("{}: {}".format(file_path, dataset_number))

            saved_data.append(data_temporary)
            dataset_number += 1

    with open(saved_file_path, 'w') as f:
        json.dump(saved_data, f, indent=4)


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, SAVED_FILE)
    