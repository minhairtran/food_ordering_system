import sys
sys.path.append("../")
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

from train.model import FoodNumberModel
from train.model import ConfirmingModel

import os
import librosa
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import librosa.display
from train.text_transform import FoodNumberTextTransform
from train.text_transform import ConfirmTextTransform

DATA_SET = "../predict/test"
SAVED_MODEL_PATH = "../train/model_food_number.h5"
# SAVED_MODEL_PATH = "../train/model_confirming.h5"

text_transform = FoodNumberTextTransform()
# text_transform = ConfirmTextTransform()

# def plot_spectrogram(Y, hop_length, y_axis="linear"):
#     plt.figure(figsize=(25, 10))
#     librosa.display.specshow(Y, x_axis="time", y_axis="mel", sr=22050)
#     plt.colorbar(format="%+2.f")
#     plt.show()

class Prediction():
    def __init__(self):
        super(Prediction, self).__init__()


    def preprocess(self, signal, n_fft=512, hop_length=384, n_mels=20,
                fmax=8000):

        # extract MFCCs
        mel_spectrogram = librosa.feature.melspectrogram(signal, n_fft=n_fft,
                                                        hop_length=hop_length, n_mels=n_mels, fmax=fmax)
        
        mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        mel_spectrogram = mel_spectrogram.T

        mel_spectrogram = np.array(mel_spectrogram[np.newaxis, ...])

        mel_spectrogram = torch.tensor(
            mel_spectrogram, dtype=torch.float).detach().requires_grad_()

        mel_spectrogram = nn.utils.rnn.pad_sequence(
            mel_spectrogram, batch_first=True).unsqueeze(1).transpose(2, 3)

        return mel_spectrogram


    def GreedyDecoder(self, output, blank_label=0, collapse_repeated=True):
        arg_maxes = torch.argmax(output, dim=2)

        decodes = []

        for i, args in enumerate(arg_maxes):
            decode = []
            for j, index in enumerate(args):
                if index != blank_label:
                    if collapse_repeated and j != 0 and index == args[j - 1]:
                        continue
                    decode.append(index.item())
            decodes.append(text_transform.int_to_text(decode))
        return decodes


    def predict(self, model, tested_audio):
        mel_spectrogram = self.preprocess(tested_audio)
        mel_spectrogram = mel_spectrogram.to(device)

        # get the predicted label
        output = model(mel_spectrogram)

        output = F.log_softmax(output, dim=2)
        predicted = self.GreedyDecoder(output)
        return predicted


if __name__ == "__main__":
    device = torch.device("cpu")

    model = FoodNumberModel(FoodNumberModel.hparams['n_cnn_layers'], FoodNumberModel.hparams['n_rnn_layers'], FoodNumberModel.hparams['rnn_dim'], FoodNumberModel.hparams['n_class'], FoodNumberModel.hparams['n_feats'], \
        FoodNumberModel.hparams['stride'], FoodNumberModel.hparams['dropout']).to(device)

    food_number_prediction = Prediction()

    # model = ConfirmingModel(ConfirmingModel.hparams['n_cnn_layers'], ConfirmingModel.hparams['n_rnn_layers'], ConfirmingModel.hparams['rnn_dim'], \
    #     ConfirmingModel.hparams['n_class'], ConfirmingModel.hparams['n_feats'], ConfirmingModel.hparams['stride'], ConfirmingModel.hparams['dropout']).to(device)

    # confirming_prediction = Prediction()

    checkpoint = torch.load(SAVED_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATA_SET)):
        for f in filenames:
            file_path = os.path.join(dirpath, f)
            signal, sample_rate = librosa.load(file_path)

            predicted_audio = food_number_prediction.predict(model, np.array(signal))
            # predicted_audio = confirming_prediction.predict(model, np.array(signal))

            print(f ,predicted_audio)

    