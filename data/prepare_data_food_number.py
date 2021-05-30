import sys
sys.path.append("../")

import json
import numpy as np
import os
import librosa
import torch
import torch.nn as nn
import torchaudio
import augment
from scipy.io import wavfile


DATASET_PATH = ["../../food_dataset/ca_kho", "../../food_dataset/ca_xot", "../../food_dataset/khoai_tay_chien", "../../food_dataset/com_heo_xi_muoi", "../../food_dataset/com_nieu", \
                    "../../food_dataset/com_tam", "../../food_dataset/com_thap_cam", "../../food_dataset/khong_biet", "../../food_dataset/rau_cai_luoc", \
                        "../../food_dataset/rau_cai_xao",  "../../food_dataset/salad_tron", "../../food_dataset/tra_hoa_cuc", "../../food_dataset/tra_sam_dua", \
                            "../../food_dataset/trung_chien"]
SAVED_FILE = ["food_data/data_ca_kho.json", "food_data/data_ca_xot.json", "food_data/data_khoai_tay_chien.json", "food_data/data_com_heo_xi_muoi.json", "food_data/data_com_nieu.json", \
                "food_data/data_com_tam.json", "food_data/data_com_thap_cam.json", "food_data/data_khong_biet.json", "food_data/data_rau_cai_luoc.json",\
                    "food_data/data_rau_cai_xao.json", "food_data/data_salad_tron.json", "food_data/data_tra_hoa_cuc.json", "food_data/data_tra_sam_dua.json", \
                        "food_data/data_trung_chien.json"]


def preprocess_dataset(dataset_path, saved_file_path):
    # mel spectrogram
    kwargs = {
        'n_fft': 512,
        'n_mels': 40,
    }
    wav_to_spec = torchaudio.transforms.MelSpectrogram(**kwargs)

    log_mel_spec = torchaudio.transforms.AmplitudeToDB()

    # spectrogram augmentation
    kwargs = {
        'freq_mask_param': 13,
        'time_mask_param': 5,
    }
    spec_augment = augment.SpectrogramAugmentation(**kwargs)

    dataset_number = 0

    for index, (data_set, save_file) in enumerate(zip(dataset_path, saved_file_path)):

        # dictionary where we'll store mapping, labels, MFCCs and filenames
        data_temporary = {
            "mel_spectrogram": [],
            "labels": []
        }

        # loop through all sub-dirs
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_set)):

            label = dirpath.split("/")[-1]

            print("\nProcessing: '{}' with save file: ".format(label, save_file))

            for f in filenames:
                file_path = os.path.join(dirpath, f)

                fs, data = wavfile.read(file_path)
                data = torch.Tensor(data.copy())
                data = data / data.abs().max()

                a = wav_to_spec(data.clone())

                x = log_mel_spec(a.clone())

                data_temporary["mel_spectrogram"].append(x.T.tolist())
                data_temporary["labels"].append(dataset_number)

                for i in range(199):
                    mel_spectrogram = np.array(spec_augment(a.clone().unsqueeze(0)).squeeze(0)).T.tolist()
                    # store data for analysed track
                    data_temporary["mel_spectrogram"].append(mel_spectrogram)
                    data_temporary["labels"].append(dataset_number)
                    print("{}: {}".format(file_path, dataset_number))
            
        print("\nSave: {}".format(save_file))
        dataset_number += 1

        # torch.save(data, save_file)
        with open(save_file, 'w') as f:
            json.dump(data_temporary, f, indent=4)


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, SAVED_FILE)