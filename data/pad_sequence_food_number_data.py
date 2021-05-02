import sys
sys.path.append("../")

import torch
import numpy as np
import json
import torch.nn as nn

DATA_FILE = ["food_number_data/data_zero.json", "food_number_data/data_one.json", "food_number_data/data_two.json", "food_number_data/data_three.json", \
    "food_number_data/data_four.json", "food_number_data/data_five.json", "food_number_data/data_six.json", "food_number_data/data_seven.json", \
        "food_number_data/data_eight.json", "food_number_data/data_nine.json", "food_number_data/data_unknown.json"]

SAVED_FILE = "food_number_data/data_set_"


def tensorize(mel_spectrogram_not_tensorized, labels_not_tensorized):
    mel_spectrogram, labels = [], []

    for spectrogram in mel_spectrogram_not_tensorized:
        mel_spectrogram.append(torch.Tensor(spectrogram))

    for label in labels_not_tensorized:
        labels.append(torch.tensor(label, dtype=torch.long))

    mel_spectrogram = nn.utils.rnn.pad_sequence(
        mel_spectrogram, batch_first=True).unsqueeze(1).transpose(2, 3)

    return mel_spectrogram, labels


if __name__ == "__main__":
    data_set = []
    all_data_length = 0

    for i, data in enumerate(DATA_FILE):
        with open(data, "r") as fp:
            temp_data_set = json.load(fp)

        data_set.append(temp_data_set)
        all_data_length += len(data_set[i]['labels'])

    average_dataset_length = int(all_data_length/len(data_set))

    for set_number in range(len(data_set)):
        saved_dataset = {
            "mel_spectrogram": [],
            "labels": []
        }
        for current_dataset_index in range(average_dataset_length):
            original_dataset_index = int(current_dataset_index/len(data_set)) + int(average_dataset_length*set_number/len(data_set)) + 1
            original_dataset_number = current_dataset_index % len(data_set)

            if (original_dataset_index >= len(data_set[original_dataset_number]["labels"])):
                break
            else:
                saved_dataset["mel_spectrogram"].append(
                    data_set[original_dataset_number]["mel_spectrogram"][original_dataset_index])
                saved_dataset["labels"].append(
                    data_set[original_dataset_number]["labels"][original_dataset_index])

        saved_dataset["mel_spectrogram"], saved_dataset["labels"] = tensorize(
            saved_dataset["mel_spectrogram"], saved_dataset["labels"])

        current_saved_file = SAVED_FILE + str(set_number) + ".pt"
        torch.save(saved_dataset, current_saved_file)

        print("Padding set success", set_number)
