import sys
sys.path.append("../")
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

from train.model import Confirming_model

import datetime
import random
import wave
import torch.nn as nn
import torch.nn.functional as F
import pyaudio
import noisereduce as nr
import torch
import numpy as np
import time
# from ctypes import *
import torchaudio


def id_generator():
    now = datetime.datetime.now()
    table_number = random.randint(0, 20)

    return now.strftime("%Y") + now.strftime("%m") + now.strftime("%d") + now.strftime("%H") + now.strftime("%M") + now.strftime("%S") + str(table_number)


FILENAME = "recorded_audios/" + id_generator() + ".wav"
SAVED_MODEL_PATH = "../train/model_confirming.h5"
# SAVED_MODEL_PATH = "../train/model_confirming_noise.h5"

CHUNKSIZE = 16000  # fixed chunk size
RATE = 16000
SAMPLE_FORMAT = pyaudio.paFloat32
CHANNELS = 1

# ERROR_HANDLER_FUNC = CFUNCTYPE(
#     None, c_char_p, c_int, c_char_p, c_int, c_char_p)


# def py_error_handler(filename, line, function, err, fmt):
#     pass

class ConfirmingPrediction():
    def __init__(self):
        super(ConfirmingPrediction, self).__init__()

    
    def preprocess(self, data):

        # mel spectrogram
        kwargs = {
            'n_fft': 512,
            'n_mels': 40
        }
        wav_to_spec = torchaudio.transforms.MelSpectrogram(**kwargs)

        log_mel_spec = torchaudio.transforms.AmplitudeToDB()


        data = torch.Tensor(data.copy())
        data = data / data.abs().max()

        mel_spectrogram = wav_to_spec(data.clone())

        log_mel_spectrogram = np.array(log_mel_spec(mel_spectrogram.clone()))

        log_mel_spectrogram = np.array(log_mel_spectrogram[np.newaxis, ...])

        log_mel_spectrogram = torch.tensor(
            log_mel_spectrogram, dtype=torch.float).detach().requires_grad_()

        return log_mel_spectrogram


    def predict(self, model, tested_audio, device=torch.device("cpu")):
        log_mel_spectrogram = self.preprocess(tested_audio)
        log_mel_spectrogram = log_mel_spectrogram.to(device)

        # get the predicted label
        output = model(log_mel_spectrogram)

        predicted = torch.argmax(output, 1).tolist()[0]

        decode = {
            0: "co",
            1: "khong",
            2: "khong biet",
        }

        # print(predicted)

        return decode[predicted]


if __name__ == "__main__":

    device = torch.device("cpu")

    model = Confirming_model(Confirming_model.hparams['n_mels'], Confirming_model.hparams['cnn_channels'], Confirming_model.hparams['cnn_kernel_size'], \
        Confirming_model.hparams['stride'], Confirming_model.hparams['gru_hidden_size'], Confirming_model.hparams['attention_hidden_size'], Confirming_model.hparams['n_classes']).to(device)

    checkpoint = torch.load(SAVED_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    confirming_prediction = ConfirmingPrediction()

    # c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    # asound = cdll.LoadLibrary('libasound.so')
    # # Set error handler
    # asound.snd_lib_error_set_handler(c_error_handler)

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

    print("Start recording...")
    start = time.time()

    # while(time.time() - start < 10):
    while(True):
        # Read chunk and load it into numpy array.
        data = stream.read(CHUNKSIZE)
        frames.append(data)
        current_window = np.frombuffer(data, dtype=np.float32)

        if(np.amax(current_window) > 0.49):
            current_window = nr.reduce_noise(audio_clip=current_window, noise_clip=noise_sample, verbose=False)
            predicted_window = np.append(predicted_window, current_window)
        else:
            current_window = nr.reduce_noise(audio_clip=current_window, noise_clip=noise_sample, verbose=False)
            if(len(predicted_window) == 0):
                #Hoi 2 anh
                noise_sample = np.frombuffer(data, dtype=np.float32)
            else:
                predicted_audio = confirming_prediction.predict(model, np.array(predicted_window))
                print(predicted_audio)
                predicted_window = np.array([])

        print((time.time() - start), np.amax(current_window))
        # time.sleep(1)

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
