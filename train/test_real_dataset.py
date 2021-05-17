import sys
sys.path.append("../")
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

# from train.model import FoodNumberModel
from train.model import KWS_model

import os
import librosa
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import librosa.display
import torchaudio
from scipy.io import wavfile


DATA_SET = "../predict/test"
# SAVED_MODEL_PATH = "../train/model_food_number.h5"
SAVED_MODEL_PATH = "../train/model_confirming.h5"

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
        # fs, data = wavfile.read(file_path)

        data, sr = librosa.load(file_path, sr=16000)

        mel_spectrogram = self.preprocess(data)
        mel_spectrogram = mel_spectrogram.to(device)

        # get the predicted label
        output = model(mel_spectrogram)

        predicted = torch.argmax(output, 1).tolist()[0]

        decode = {
            0: "co",
            1: "khong",
            2: "khong biet",
        }

        return decode[predicted]


if __name__ == "__main__":
    device = torch.device("cpu")

    # model = FoodNumberModel(FoodNumberModel.hparams['n_cnn_layers'], FoodNumberModel.hparams['n_rnn_layers'], FoodNumberModel.hparams['rnn_dim'], FoodNumberModel.hparams['n_class'], FoodNumberModel.hparams['n_feats'], \
    #     FoodNumberModel.hparams['stride'], FoodNumberModel.hparams['dropout']).to(device)

    # food_number_prediction = Prediction()

    model = KWS_model(KWS_model.hparams['n_mels'], KWS_model.hparams['cnn_channels'], KWS_model.hparams['cnn_kernel_size'], \
        KWS_model.hparams['gru_hidden_size'], KWS_model.hparams['attention_hidden_size'], KWS_model.hparams['n_classes']).to(device)

    confirming_prediction = Prediction()

    checkpoint = torch.load(SAVED_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATA_SET)):
        for f in filenames:
            file_path = os.path.join(dirpath, f)

            # predicted_audio = food_number_prediction.predict(model, np.array(signal))
            predicted_audio = confirming_prediction.predict(model, file_path)

            print(f ,predicted_audio)

    