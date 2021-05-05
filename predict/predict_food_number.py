import sys
sys.path.append("../")
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

from train.model import FoodNumberModel

import datetime
import random
import wave
import torch.nn as nn
import torch.nn.functional as F
import pyaudio
import os
import noisereduce as nr
import librosa
import torch
import numpy as np
import time
from ctypes import *

def id_generator():
    now = datetime.datetime.now()
    table_number = random.randint(0, 20)

    return now.strftime("%Y") + now.strftime("%m") + now.strftime("%d") + now.strftime("%H") + now.strftime("%M") + now.strftime("%S") + str(table_number)


FILENAME = "recorded_audios/" + id_generator() + ".wav"
SAVED_MODEL_PATH = "../train/model_food_number.h5"
CHUNKSIZE = 22050  # fixed chunk size
RATE = 22050
SAMPLE_FORMAT = pyaudio.paFloat32
CHANNELS = 2

ERROR_HANDLER_FUNC = CFUNCTYPE(
    None, c_char_p, c_int, c_char_p, c_int, c_char_p)


def py_error_handler(filename, line, function, err, fmt):
    pass

class FoodNumberPrediction():
    def __init__(self):
        super(FoodNumberPrediction, self).__init__()


    def preprocess(self, signal, n_fft=512, hop_length=384, n_mels=20,
                fmax=8000):

        # extract MFCCs
        mel_spectrogram = librosa.feature.melspectrogram(signal, n_fft=n_fft,
                                                        hop_length=hop_length, n_mels=n_mels, fmax=fmax)

        mel_spectrogram = np.array(mel_spectrogram[..., np.newaxis])

        mel_spectrogram = torch.tensor(
            mel_spectrogram.T, dtype=torch.float).detach().requires_grad_()

        mel_spectrogram = nn.utils.rnn.pad_sequence(
            mel_spectrogram, batch_first=True).unsqueeze(1).transpose(2, 3)

        return mel_spectrogram


    def decoder(self, output):
        arg_maxes = torch.argmax(output, dim=1).tolist()
        
        decode = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "unknown"
        }

        return decode.get(arg_maxes[0], "unknown")


    def predict(self, model, tested_audio):
        mel_spectrogram = self.preprocess(tested_audio)
        mel_spectrogram = mel_spectrogram.to(device)

        # get the predicted label
        output = model(mel_spectrogram)

        output = F.log_softmax(output, dim=1)
        predicted = self.decoder(output)
        return predicted


if __name__ == "__main__":
    device = torch.device("cpu")

    model = FoodNumberModel(FoodNumberModel.hparams['n_cnn_layers'], FoodNumberModel.hparams['n_rnn_layers'], FoodNumberModel.hparams['rnn_dim'], FoodNumberModel.hparams['n_class'], FoodNumberModel.hparams['n_feats'], \
        FoodNumberModel.hparams['stride'], FoodNumberModel.hparams['dropout']).to(device)

    checkpoint = torch.load(SAVED_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    food_number_prediction = FoodNumberPrediction()

    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so')
    # Set error handler
    asound.snd_lib_error_set_handler(c_error_handler)

    # initialize portaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=SAMPLE_FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

    # noise window
    data = stream.read(CHUNKSIZE)
    noise_sample = np.frombuffer(data, dtype=np.float32)
    audio_buffer = []
    frames = []
    predicted_window = np.array([])


    print("Start recording...")
    start = time.time()

    # while(time.time() - start < 60):
    while(True):
        # Read chunk and load it into numpy array.
        data = stream.read(CHUNKSIZE)
        frames.append(data)
        current_window = np.frombuffer(data, dtype=np.float32)

        current_window = nr.reduce_noise(audio_clip=current_window, noise_clip=noise_sample, verbose=False)

        if(np.amax(current_window) > 0.9):
            predicted_window = np.append(predicted_window, current_window)
        else:
            if(len(predicted_window) == 0):
                #Hoi 2 anh
                noise_sample = np.frombuffer(data, dtype=np.float32)
            else:
                predicted_audio = food_number_prediction.predict(model, np.array(current_window))
                print(predicted_audio)
                predicted_window = np.array([])

        print((time.time() - start), np.amax(current_window))

    # close stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
