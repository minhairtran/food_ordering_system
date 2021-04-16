import sys
sys.path.append("../")

import json
import numpy as np
from train.text_transform import FoodNumberTextTransform
import os
import librosa
import torch
import torch.nn as nn


DATASET_PATH = ["../../food_number_dataset/zero", "../../food_number_dataset/one", "../../food_number_dataset/two", "../../food_number_dataset/three", "../../food_number_dataset/four",
                "../../food_number_dataset/five", "../../food_number_dataset/six", "../../food_number_dataset/seven", "../../food_number_dataset/eight", "../../food_number_dataset/nine", "../../food_number_dataset/noise"]
SAVED_FILE = ["food_number_data/data_zero.json", "food_number_data/data_one.json", "food_number_data/data_two.json", "food_number_data/data_three.json", "food_number_data/data_four.json",
              "food_number_data/data_five.json", "food_number_data/data_six.json", "food_number_data/data_seven.json", "food_number_data/data_eight.json", "food_number_data/data_nine.json", "food_number_data/data_noise.json"]


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

                text_transform = FoodNumberTextTransform()

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
