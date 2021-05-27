import sys
sys.path.append("../")

import torch
import numpy as np
import json
import torch.nn as nn

DATA_FILE = ["food_data/data_ca_kho.json", "food_data/data_ca_xot.json", "food_data/data_khoai_tay_chien.json", "food_data/data_com_heo_xi_muoi.json", "food_data/data_com_nieu.json", \
                "food_data/data_com_tam.json", "food_data/data_com_thap_cam.json", "food_data/data_khong_biet.json", "food_data/data_rau_cai_luoc.json",\
                    "food_data/data_rau_cai_xao.json", "food_data/data_salad_tron.json", "food_data/data_tra_hoa_cuc.json", "food_data/data_tra_sam_dua.json", \
                        "food_data/data_trung_chien.json"]

SAVED_FILE = "food_data/data_set_"


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
            original_dataset_index = int(current_dataset_index/len(data_set)) + int(average_dataset_length*set_number/len(data_set))
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
        print("Padding set success {}, length {}".format(set_number, len(saved_dataset["labels"])))