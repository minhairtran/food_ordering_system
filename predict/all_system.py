import sys
sys.path.append("../")
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

from train.model import ConfirmingModel, FoodNumberModel
from predict_confirming import ConfirmingPrediction
from predict_food_number import FoodNumberPrediction

import datetime
import random
import wave
import torch.nn as nn
import torch.nn.functional as F
import pyaudio
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

SAVE_AUDIO_FILE_PATH = "recorded_audios/" + id_generator() + ".wav"
CONFIRMING_MODEL_PATH = "../train/model_confirming.h5"
FOOD_NUMBER_MODEL_PATH = "../train/model_food_number.h5"

CHUNKSIZE = 16000  # fixed chunk size
RATE = 16000
SAMPLE_FORMAT = pyaudio.paFloat32
CHANNELS = 2

ERROR_HANDLER_FUNC = CFUNCTYPE(
    None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass

if __name__ == "__main__":
    device = torch.device("cpu")

    # Confirming model initialization
    confirming_model = ConfirmingModel(ConfirmingModel.hparams['n_cnn_layers'], ConfirmingModel.hparams['n_rnn_layers'], ConfirmingModel.hparams['rnn_dim'], ConfirmingModel.hparams['n_class'], ConfirmingModel.hparams['n_feats'], \
        ConfirmingModel.hparams['stride'], ConfirmingModel.hparams['dropout']).to(device)

    confirming_model_checkpoint = torch.load(CONFIRMING_MODEL_PATH, map_location=device)
    confirming_model.load_state_dict(confirming_model_checkpoint)
    confirming_model.eval()

    confirming_prediction = ConfirmingPrediction()

    # Food model initialization 
    food_number_model = FoodNumberModel(FoodNumberModel.hparams['n_cnn_layers'], FoodNumberModel.hparams['n_rnn_layers'], FoodNumberModel.hparams['rnn_dim'], FoodNumberModel.hparams['n_class'], FoodNumberModel.hparams['n_feats'], \
        FoodNumberModel.hparams['stride'], FoodNumberModel.hparams['dropout']).to(device)

    food_number_checkpoint = torch.load(SAVED_MODEL_PATH, map_location=device)
    food_number_model.load_state_dict(food_number_checkpoint)
    food_number_model.eval()

    food_number_prediction = FoodNumberPrediction()

    # Handle streaming error
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)

    # initialize portaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=SAMPLE_FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

    # noise window
    data = stream.read(CHUNKSIZE)
    noise_sample = np.frombuffer(data, dtype=np.float32)
    # loud_threshold = np.mean(np.abs(noise_sample)) * 10
    audio_buffer = []
    frames = []
    predicted_window = np.array([])

    

