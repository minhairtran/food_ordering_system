import sys
sys.path.append("../")
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

from train.model import Food_model
# from train.model import Confirming_model

import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchaudio
from scipy.io import wavfile


DATA_SET = "../predict/test"
SAVED_MODEL_PATH = "../train/model_food_number.h5"
# SAVED_MODEL_PATH = "../train/model_confirming.h5"

# def plot_spectrogram(Y, hop_length, y_axis="linear"):
#     plt.figure(figsize=(25, 10))
#     librosa.display.specshow(Y, x_axis="time", y_axis="mel", sr=22050)
#     plt.colorbar(format="%+2.f")
#     plt.show()

class Prediction():
    def __init__(self):
        super(Prediction, self).__init__()


    def preprocess(self, data):
        # mel spectrogram
        kwargs = {
            'n_fft': 512,
            'n_mels': 40
        }
        wav_to_spec = torchaudio.transforms.MelSpectrogram(**kwargs)

        data = torch.Tensor(data.copy())
        data = data / data.abs().max()

        mel_spectrogram = np.array(wav_to_spec(data.clone()))

        mel_spectrogram = np.array(mel_spectrogram[np.newaxis, ...])

        mel_spectrogram = torch.tensor(
            mel_spectrogram, dtype=torch.float).detach().requires_grad_()

        return mel_spectrogram


    def predict(self, model, file_path):
        fs, data = wavfile.read(file_path)

        mel_spectrogram = self.preprocess(data)
        mel_spectrogram = mel_spectrogram.to(device)

        # get the predicted label
        output = model(mel_spectrogram)

        predicted = torch.argmax(output, 1).tolist()[0]

        # decode = {
        #     0: "co",
        #     1: "khong",
        #     2: "khong_biet",
        # }

        decode = {
            0: "ca_kho",
            1: "ca_xot",
            2: "com_ga",
            3: "com_heo_xi_muoi",
            4: "com_nieu",
            5: "com_tam",
            6: "com_thap_cam",
            7: "khong_biet",
            8: "rau_muong_luoc",
            9: "rau_muong_xao",
            10: "salad_tron",
            11: "tra_hoa_cuc",
            12: "tra_sam_dua",
            13: "trung_chien",
        }

        return decode[predicted]


if __name__ == "__main__":
    device = torch.device("cpu")

    model = Food_model(Food_model.hparams['n_mels'], Food_model.hparams['cnn_channels'], Food_model.hparams['cnn_kernel_size'], \
        Food_model.hparams['gru_hidden_size'], Food_model.hparams['attention_hidden_size'], Food_model.hparams['n_classes']).to(device)

    food_number_prediction = Prediction()

    # model = Confirming_model(Confirming_model.hparams['n_mels'], Confirming_model.hparams['cnn_channels'], Confirming_model.hparams['cnn_kernel_size'], \
    #     Confirming_model.hparams['gru_hidden_size'], Confirming_model.hparams['attention_hidden_size'], Confirming_model.hparams['n_classes']).to(device)

    # confirming_prediction = Prediction()

    checkpoint = torch.load(SAVED_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATA_SET)):
        for f in filenames:
            file_path = os.path.join(dirpath, f)

            predicted_audio = food_number_prediction.predict(model, file_path)
            # predicted_audio = confirming_prediction.predict(model, file_path)

            print(f ,predicted_audio)

    