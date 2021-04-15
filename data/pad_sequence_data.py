import sys
sys.path.append("../")

import torch
import numpy as np
import json
import torch.nn as nn


# DATA_FILE = ["confirming_data/data_yes.json", "confirming_data/data_no.json"]
DATA_FILE = ["food_number_data/data_zero.json", "food_number_data/data_one.json", "food_number_data/data_two.json", "food_number_data/data_three.json", \
    "food_number_data/data_four.json", "food_number_data/data_five.json", "food_number_data/data_six.json", "food_number_data/data_seven.json", \
        "food_number_data/data_eight.json", "food_number_data/data_nine.json"]

SAVED_FILE = "food_number_data/data_set_"

DATA_NOISE = "food_number_data/data_noise.json"


def tensorize(mel_spectrogram_not_tensorized, labels_not_tensorized):
    mel_spectrogram, labels = [], []

    for spectrogram in mel_spectrogram_not_tensorized:
        mel_spectrogram.append(torch.Tensor(spectrogram))

    for label in labels_not_tensorized:
        labels.append(torch.Tensor(label))

    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    mel_spectrogram = nn.utils.rnn.pad_sequence(
        mel_spectrogram, batch_first=True).transpose(1, 2)

    return mel_spectrogram, labels


if __name__ == "__main__":
    data_set = []
    all_data_length = 0

    for i, data in enumerate(DATA_FILE):
        with open(data, "r") as fp:
            temp_data_set = json.load(fp)

        data_set.append(temp_data_set)
        all_data_length += len(data_set[i]['labels'])

    with open(DATA_NOISE, "r") as fp:
        data_noise = json.load(fp)

    average_dataset_length = int(all_data_length/len(data_set))

    for set_number in range(len(data_set)):
        saved_dataset = {
            "label_lengths": [],
            "mel_spectrogram": [],
            "labels": [],
            "input_lengths": []
        }
        for current_dataset_index in range(average_dataset_length):
            original_dataset_index = int(current_dataset_index/len(data_set))
            original_dataset_number = current_dataset_index % len(data_set)

            if (original_dataset_index >= len(data_set[original_dataset_number]["label_lengths"])):
                break
            else:
                saved_dataset["mel_spectrogram"].append(
                    data_set[original_dataset_number]["mel_spectrogram"][original_dataset_index])
                saved_dataset["label_lengths"].append(
                    data_set[original_dataset_number]["label_lengths"][original_dataset_index])
                saved_dataset["labels"].append(
                    data_set[original_dataset_number]["labels"][original_dataset_index])
                saved_dataset["input_lengths"].append(
                    data_set[original_dataset_number]["input_lengths"][original_dataset_index])

        for data_noise_index in range(6):
            saved_dataset["mel_spectrogram"].append(
                data_noise["mel_spectrogram"][data_noise_index])
            saved_dataset["label_lengths"].append(
                data_noise["label_lengths"][data_noise_index])
            saved_dataset["labels"].append(
                data_noise["labels"][data_noise_index])
            saved_dataset["input_lengths"].append(
                data_noise["input_lengths"][data_noise_index])

        saved_dataset["mel_spectrogram"], saved_dataset["labels"] = tensorize(
            saved_dataset["mel_spectrogram"], saved_dataset["labels"])
        saved_dataset["input_lengths"] = torch.Tensor(
            saved_dataset["input_lengths"])
        saved_dataset["label_lengths"] = torch.Tensor(
            saved_dataset["label_lengths"])

        current_saved_file = SAVED_FILE + str(set_number) + ".pt"
        torch.save(saved_dataset, current_saved_file)

        print("Padding set success", set_number)
