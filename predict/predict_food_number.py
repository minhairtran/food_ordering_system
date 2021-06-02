import sys
sys.path.append("../")
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

from train.model import Food_model

import datetime
import random
import wave
import pyaudio
import noisereduce as nr
import torch
import numpy as np
import time
# import ctypes_callable
import torchaudio


def id_generator():
    now = datetime.datetime.now()
    table_number = random.randint(0, 20)

    return now.strftime("%Y") + now.strftime("%m") + now.strftime("%d") + now.strftime("%H") + now.strftime("%M") + now.strftime("%S") + str(table_number)


FILENAME = "recorded_audios/" + id_generator() + ".wav"
SAVED_MODEL_PATH = "../train/model_food_number_13_5.h5"
CHUNKSIZE = 16000  # fixed chunk size
RATE = 16000
SAMPLE_FORMAT = pyaudio.paFloat32
CHANNELS = 1


# ERROR_HANDLER_FUNC = CFUNCTYPE(
#     None, c_char_p, c_int, c_char_p, c_int, c_char_p)


# def py_error_handler(filename, line, function, err, fmt):
#     pass

class FoodNumberPrediction():
    def __init__(self):
        super(FoodNumberPrediction, self).__init__()


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

        log_mel_spectrogram = np.array(log_mel_spectrogram[np.newaxis, np.newaxis, ...])

        log_mel_spectrogram = torch.tensor(
            log_mel_spectrogram, dtype=torch.float).detach().requires_grad_()

        return log_mel_spectrogram

    def predict(self, model, tested_audio, device=torch.device("cpu")):
        mel_spectrogram = self.preprocess(tested_audio)
        mel_spectrogram = mel_spectrogram.to(device)

        # get the predicted label
        output = model(mel_spectrogram)

        predicted = torch.argmax(output, 1).tolist()[0]

        decode = {
            0: "ca_kho",
            1: "ca_xot",
            2: "khoai_tay_chien",
            3: "com_heo_xi_muoi",
            4: "com_nieu",
            5: "com_tam",
            6: "com_thap_cam",
            7: "khong_biet",
            8: "rau_cai_luoc",
            9: "rau_cai_xao",
            10: "salad_tron",
            11: "tra_hoa_cuc",
            12: "tra_sam_dua",
            13: "trung_chien",
        }

        # print(predicted)

        return decode[predicted]


if __name__ == "__main__":
    device = torch.device("cpu")

    model = Food_model(Food_model.hparams['n_mels'], Food_model.hparams['cnn_channels'], Food_model.hparams['cnn_kernel_size'], \
        Food_model.hparams['stride'], Food_model.hparams['gru_hidden_size'], Food_model.hparams['attention_hidden_size'], Food_model.hparams['n_classes']).to(device)

    checkpoint = torch.load(SAVED_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    food_number_prediction = FoodNumberPrediction()

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

        if(np.amax(current_window) > 0.01):
            current_window = nr.reduce_noise(audio_clip=current_window, noise_clip=noise_sample, verbose=False)
            predicted_window = np.append(predicted_window, current_window)
        else:
            if(len(predicted_window) == 0):
                #Hoi 2 anh
                noise_sample = np.frombuffer(data, dtype=np.float32)
            else:
                predicted_audio = food_number_prediction.predict(model, np.array(predicted_window))
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
