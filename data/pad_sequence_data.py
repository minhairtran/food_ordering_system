import sys
sys.path.append("../")

import torch
import numpy as np

YES_DATASET_PATH = "confirming_data/data_yes.pt"
# NO_DATASET_PATH = "confirming_data/data_no.pt"
NOISE_DATASET_PATH = "confirming_data/data_noise.pt"
SAVED_FILE = "confirming_data/data.pt"

def pad_sequence(first_set, second_set):
    first_set["mel_spectrogram"].tolist().append(second_set["mel_spectrogram"].tolist())
    first_set["mel_spectrogram"] = nn.utils.rnn.pad_sequence(first_set["mel_spectrogram"].tolist(), batch_first=True)

    first_set["labels"].tolist().append(second_set["labels"].tolist())
    first_set["labels"] = nn.utils.rnn.pad_sequence(first_set["labels"].tolist(), batch_first=True)

    return first_set["mel_spectrogram"], first_set["labels"]

if __name__ == "__main__":
    data = {
        "label_lengths": [],
        "mel_spectrogram": [],
        "labels": [],
        "input_lengths": []
    }

    yes_data = torch.load(YES_DATASET_PATH)
    noise_data = torch.load(NOISE_DATASET_PATH)
    data["mel_spectrogram"], data["labels"] = pad_sequence(yes_data, noise_data)

    data["label_lengths"] = torch.Tensor(yes_data["label_lengths"].tolist().append(noise_data["label_lengths"].tolist()))

    data["input_lengths"] = torch.Tensor(yes_data["input_lengths"].tolist().append(noise_data["input_lengths"].tolist()))

    torch.save(data, SAVED_FILE)

