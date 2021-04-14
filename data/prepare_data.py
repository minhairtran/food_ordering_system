import sys
sys.path.append("../")

import torch.nn as nn
import torch
import librosa
import os
from train.text_transform import ConfirmTextTransform
import numpy as np

DATASET_PATH = ["../../confirming_dataset/yes", "../../confirming_dataset/no", "../../confirming_dataset/noise"]
SAVED_FILE = ["confirming_data/data_yes.pt", "confirming_data/data_no.pt","confirming_data/data_noise.pt"]

def tensorize(mel_spectrogram_not_tensorized, labels_not_tensorized):
    mel_spectrogram, labels = [], []

    for spectrogram in mel_spectrogram_not_tensorized:
        mel_spectrogram.append(torch.Tensor(spectrogram))

    for label in labels_not_tensorized:
        labels.append(torch.Tensor(label))

    return mel_spectrogram, labels

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

            if(label=="noise"):
                label = "                                                           "
            print("\nProcessing: '{}'".format(label))

            mel_spectrogram_not_tensorized, labels_not_tensorized = [], []

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
                mel_spectrogram_not_tensorized.append(mel_spectrogram.T.tolist())
                labels_not_tensorized.append(added_label)
                data["input_lengths"].append(mel_spectrogram.T.shape[0]//2)
                data["label_lengths"].append(len(label))
                print("{}: {}".format(file_path, i-1))

            mel_spectrogram_tensorized, labels_tensorized = tensorize(mel_spectrogram_not_tensorized, labels_not_tensorized)

            data["mel_spectrogram"] = nn.utils.rnn.pad_sequence(mel_spectrogram_tensorized, batch_first=True).transpose(1, 2)
            data["labels"] = nn.utils.rnn.pad_sequence(labels_tensorized, batch_first=True)
            data["input_lengths"] = torch.Tensor(data["input_lengths"])
            data["label_lengths"] = torch.Tensor(data["label_lengths"])

        torch.save(data, save_file)


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, SAVED_FILE)