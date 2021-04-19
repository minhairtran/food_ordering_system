import sys
sys.path.append("../")

import torch
import numpy as np
import json
import torch.nn as nn


DATA_FILE = "confirming_data/data.json"
SAVED_FILE = "confirming_data/data.pt"

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
    saved_dataset = []

    with open(DATA_FILE, "r") as fp:
        temp_data_set = json.load(fp)

    for i in range(len(temp_data_set)):
        data_set.append(temp_data_set[i])
        all_data_length += len(temp_data_set[i]['labels'])

    average_dataset_length = int(all_data_length/len(data_set))

    for set_number in range(len(data_set)):
        saved_dataset_temp = {
            "label_lengths": [],
            "mel_spectrogram": [],
            "labels": [],
            "input_lengths": []
        }
        for current_dataset_index in range(average_dataset_length):
            original_dataset_index = int(current_dataset_index/len(data_set)) + average_dataset_length*set_number/len(data_set)
            original_dataset_number = current_dataset_index % len(data_set)

            if (original_dataset_index >= len(data_set[original_dataset_number]["label_lengths"])):
                break
            else:
                saved_dataset_temp["mel_spectrogram"].append(
                    data_set[original_dataset_number]["mel_spectrogram"][original_dataset_index])
                saved_dataset_temp["label_lengths"].append(
                    data_set[original_dataset_number]["label_lengths"][original_dataset_index])
                saved_dataset_temp["labels"].append(
                    data_set[original_dataset_number]["labels"][original_dataset_index])
                saved_dataset_temp["input_lengths"].append(
                    data_set[original_dataset_number]["input_lengths"][original_dataset_index])

        saved_dataset_temp["mel_spectrogram"], saved_dataset_temp["labels"] = tensorize(
            saved_dataset_temp["mel_spectrogram"], saved_dataset_temp["labels"])
        saved_dataset_temp["input_lengths"] = torch.Tensor(
            saved_dataset_temp["input_lengths"])
        saved_dataset_temp["label_lengths"] = torch.Tensor(
            saved_dataset_temp["label_lengths"])

        saved_dataset.append(saved_dataset_temp)

        print("Padding set success", set_number)

    torch.save(saved_dataset, SAVED_FILE)
