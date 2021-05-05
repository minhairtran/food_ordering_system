import sys
sys.path.append("../")
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

from train.model import SpeechRecognitionModel

import datetime
import random
import wave
from train.text_transform import FoodNumberTextTransform
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

def preprocess(signal, n_fft=512, hop_length=384, n_mels=128,
               fmax=8000):

    # extract MFCCs
    mel_spectrogram = librosa.feature.melspectrogram(signal, n_fft=n_fft,
                                                     hop_length=hop_length, n_mels=n_mels, fmax=fmax)

    mel_spectrogram = np.array(mel_spectrogram[..., np.newaxis])

    mel_spectrogram = torch.tensor(
        mel_spectrogram.T, dtype=torch.float).detach().requires_grad_()

    mel_spectrogram = nn.utils.rnn.pad_sequence(
        mel_spectrogram, batch_first=True).transpose(1, 2)

    return mel_spectrogram


def decoder(output):
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


def predict(model, tested_audio):
    mel_spectrogram = preprocess(tested_audio)
    mel_spectrogram = mel_spectrogram.to(device)

    # get the predicted label
    output = model(mel_spectrogram)

    output = F.log_softmax(output, dim=2)
    predicted = decoder(output)
    return predicted


if __name__ == "__main__":

    # use_cuda = torch.cuda.is_available()
    # torch.manual_seed(7)
    # device = torch.device("cuda" if use_cuda else "cpu")

    device = torch.device("cpu")

    model = SpeechRecognitionModel(SpeechRecognitionModel.hparams['n_rnn_layers'], SpeechRecognitionModel.hparams['rnn_dim'], \
        16, SpeechRecognitionModel.hparams['n_feats'], SpeechRecognitionModel.hparams['stride'], SpeechRecognitionModel.hparams['dropout']).to(device)

    checkpoint = torch.load(SAVED_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

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
    loud_threshold = np.mean(np.abs(noise_sample)) * 10
    audio_buffer = []
    frames = []
    near = 0

    print("Start recording...")
    start = time.time()

    while(True):
        # Read chunk and load it into numpy array.
        data = stream.read(CHUNKSIZE)
        frames.append(data)
        current_window = np.frombuffer(data, dtype=np.float32)

        current_window = nr.reduce_noise(audio_clip=current_window, noise_clip=noise_sample, verbose=False)

        if(np.mean(np.abs(current_window)) > 0.009 and np.amax(current_window) > 0.5):
            predicted_audio = predict(model, np.array(current_window))
            print(predicted_audio)
        else:
            pass

        print((time.time() - start), np.mean(np.abs(current_window)), np.amax(current_window))

        time.sleep(1)

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
