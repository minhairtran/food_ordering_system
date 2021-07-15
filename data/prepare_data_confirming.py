#Preprocess confirming dataset 

import sys
sys.path.append("../")

import torch
import os
import numpy as np
import json
import torchaudio
import augment
from scipy.io import wavfile

DATASET_PATH = ["../../confirming_dataset/co", "../../confirming_dataset/khong"]
SAVED_FILE = "confirming_data/data.json"

def preprocess_dataset(dataset_path, saved_file_path):
    torch.set_printoptions(edgeitems=1000)

    saved_data = []
    dataset_number = 0

    # Log Mel Spec params
    kwargs = {
        'n_fft': 512,
        'n_mels': 40,
    }
    wav_to_spec = torchaudio.transforms.MelSpectrogram(**kwargs)

    log_mel_spec = torchaudio.transforms.AmplitudeToDB()

    # SpecAugment param
    kwargs = {
        'freq_mask_param': 14,
        'time_mask_param': 10,
    }
    spec_augment = augment.SpectrogramAugmentation(**kwargs)

    for data_set in dataset_path:

        # dictionary where we'll store log mel and labels
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
                # Normalized raw data
                data = data / data.abs().max()

                mel_spectrogram = wav_to_spec(data.clone())

                log_mel_spectrogram = log_mel_spec(mel_spectrogram.clone())

                # Adding original log mel spec to list
                data_temporary["mel_spectrogram"].append(log_mel_spectrogram.T.tolist())
                data_temporary["labels"].append(dataset_number)

                # Adding generated log mel spec by SpecAugment to list
                for i in range(99):
                    mel_spectrogram = np.array(spec_augment(log_mel_spectrogram.clone().unsqueeze(0)).squeeze(0)).T.tolist()

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
    