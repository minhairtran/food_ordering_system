import torch
import numpy as np
import json
import torch.nn as nn

import sys
sys.path.append("../")


DATA_FILE = ["confirming_data/data_yes.json", "confirming_data/data_no.json"]
SAVED_FILE = "confirming_data/data_set_"


def tensorize(mel_spectrogram_not_tensorized, labels_not_tensorized):
    mel_spectrogram, labels = [], []

    for spectrogram in mel_spectrogram_not_tensorized:
        mel_spectrogram.append(torch.Tensor(spectrogram))

    for label in labels_not_tensorized:
        labels.append(torch.Tensor(label))

    print(mel_spectrogram.size())
    
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    mel_spectrogram = nn.utils.rnn.pad_sequence(
        mel_spectrogram, batch_first=True).transpose(1, 2)

    return mel_spectrogram, labels


if __name__ == "__main__":
    data_set = []
    all_data_length = 0

    for i, data in enumerate(DATA_FILE):
        with open(data, "r") as fp:
            data_set.append(json.load(fp))
            all_data_length += len(data_set[i]['labels'])

    average_dataset_length = int(all_data_length/len(data_set))

    for set_number in range(len(data_set)):
        saved_dataset = {
            "label_lengths": [],
            "mel_spectrogram": [],
            "labels": [],
            "input_lengths": []
        }
        for current_dataset_index in range(average_dataset_length):
            if (int(current_dataset_index/len(data_set)) >= len(dataset[current_dataset_index % len(data_set)["label_lengths"]])):
                break
            else:
                saved_dataset["label_lengths"].append(dataset[current_dataset_index % len(
                    data_set)["label_lengths"][int(current_dataset_index/len(data_set))]])
                saved_dataset["mel_spectrogram"].append(dataset[current_dataset_index % len(
                    data_set)["mel_spectrogram"][int(current_dataset_index/len(data_set))]])
                saved_dataset["labels"].append(dataset[current_dataset_index % len(
                    data_set)["labels"][int(current_dataset_index/len(data_set))]])
                saved_dataset["input_lengths"].append(dataset[current_dataset_index % len(
                    data_set)["input_lengths"][int(current_dataset_index/len(data_set))]])

        saved_dataset["mel_spectrogram"], saved_dataset["labels"] = tensorize(saved_dataset["mel_spectrogram"], saved_dataset["labels"])
        saved_dataset["input_lengths"] = torch.Tensor(saved_dataset["input_lengths"])
        saved_dataset["label_lengths"] = torch.Tensor(saved_dataset["label_lengths"])

        current_saved_file = SAVED_FILE + str(set_number) + ".pt"
        torch.save(saved_dataset, current_saved_file)

        print("Padding set success " + str(set_number))

