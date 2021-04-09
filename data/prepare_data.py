import sys
sys.path.append("../")


import librosa
import os
import json
from train.TextTransform import TextTransform

DATASET_PATH = "../../confirming_dataset"
JSON_PATH = "confirming_data/data.json"


def preprocess_dataset(dataset_path, json_path, n_mels=128, n_fft=512, hop_length=384):
    """Extracts MFCCs from music dataset and saves them into a json file.
    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :return:
    """

    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        "mel_spectrogram": [],
        "labels": [],
        "label_lengths": [],
        "input_lengths": []
    }

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:

            label = dirpath.split("/")[-1]

            if(label=="noise"):
                label = ""
            print("\nProcessing: '{}'".format(label))

            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)

                # extract MFCCs
                mel_spectrogram = librosa.feature.melspectrogram(signal, sample_rate, n_mels=n_mels, n_fft=n_fft,
                                                hop_length=hop_length)
                
                text_transform = TextTransform()

                added_label = text_transform.text_to_int(label)

                # store data for analysed track
                data["mel_spectrogram"].append(mel_spectrogram.T.tolist())
                data["labels"].append(added_label)
                data["input_lengths"].append(mel_spectrogram.T.shape[0]//2)
                data["label_lengths"].append(len(label))
                print("{}: {}".format(file_path, i-1))

    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)