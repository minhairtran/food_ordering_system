import sys
sys.path.append("../")

import torch.nn as nn
import torch
import librosa
import os
from train.text_transform import ConfirmTextTransform
import numpy as np
import json

DATASET_PATH = ["../../confirming_dataset/yes", "../../confirming_dataset/no", "../../confirming_dataset/noise"]
SAVED_FILE = ["confirming_data/data_yes.json", "confirming_data/data_no.json", "confirming_data/data_noise.json"]

# DATASET_PATH = ["../../food_number_dataset/yes", "../../food_number_dataset/no"]
# SAVED_FILE = ["food_number_dataset/data_yes.json", "food_number_dataset/data_no.json"]

def preprocess_dataset(dataset_path, saved_file_path, n_mels=128, n_fft=512, hop_length=384):
    for index, (data_set, save_file) in enumerate(zip(dataset_path, saved_file_path)):

        # dictionary where we'll store mapping, labels, MFCCs and filenames
        data = {
            "label_lengths": [],
            "mel_spectrogram": [],
            "labels": [],
            "input_lengths": []
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
                                                hop_length=hop_length)
                
                text_transform = ConfirmTextTransform()

                added_label = text_transform.text_to_int(label)

                # store data for analysed track
                data["mel_spectrogram"].append(mel_spectrogram.T.tolist())
                data["labels"].append(added_label)
                data["input_lengths"].append(mel_spectrogram.T.shape[0]//2)
                data["label_lengths"].append(len(label))
                print("{}: {}".format(file_path, i-1))

        # torch.save(data, save_file)
        with open(save_file, 'w') as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, SAVED_FILE)