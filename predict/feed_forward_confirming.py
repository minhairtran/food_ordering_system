import sys
sys.path.append("/home/minhair/Desktop/food_ordering_system/")
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

import datetime
import random
import wave
from food_ordering_system.train.TextTransform import TextTransform
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


FILENAME = "/home/minhair/Desktop/food_ordering_system/food_ordering_system/predict/recorded_audios/" + id_generator() + ".wav"
SAVED_MODEL_PATH = "/home/minhair/Desktop/food_ordering_system/food_ordering_system/train/model_confirming.h5"
CHUNKSIZE = 22050  # fixed chunk size
RATE = 22050
SAMPLE_FORMAT = pyaudio.paFloat32
CHANNELS = 2

ERROR_HANDLER_FUNC = CFUNCTYPE(
    None, c_char_p, c_int, c_char_p, c_int, c_char_p)


def py_error_handler(filename, line, function, err, fmt):
    pass


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        # if(x[0][0][0].shape!=torch.Size([64])):
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        # (batch, channel, feature, time)
        return x.transpose(2, 3).contiguous()


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels,
                              kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels,
                              kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):

    def __init__(self, n_cnn_layers=3, n_rnn_layers=5, rnn_dim=512, n_class=6, n_feats=128, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        # cnn for extracting heirachal features
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1,
                        dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2],
                   sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x


def preprocess(signal, n_fft=512, hop_length=384, n_mels=128,
               fmax=8000):

    # extract MFCCs
    mel_spectrogram = librosa.feature.melspectrogram(signal, n_fft=n_fft,
                                                     hop_length=hop_length, n_mels=n_mels, fmax=fmax)

    mel_spectrogram = np.array(mel_spectrogram[..., np.newaxis, np.newaxis])

    mel_spectrogram = torch.tensor(
        mel_spectrogram.T, dtype=torch.float).detach().requires_grad_()

    mel_spectrogram = nn.utils.rnn.pad_sequence(
        mel_spectrogram, batch_first=True).transpose(2, 3)

    return mel_spectrogram


def decoder(output, blank_label=0, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)

    decodes = []

    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        text_transform = TextTransform()
        decodes.append(text_transform.int_to_text(decode))
    return text_transform.list_to_string(decodes)


def predict(model, tested_audio):
    mel_spectrogram = preprocess(tested_audio)
    mel_spectrogram = mel_spectrogram.to(device)

    # get the predicted label
    output = model(mel_spectrogram)

    output = F.log_softmax(output, dim=2)
    predicted = decoder(output)
    return predicted


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = SpeechRecognitionModel().to(device)
    checkpoint = torch.load(SAVED_MODEL_PATH)
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
    data = stream.read(10000)
    noise_sample = np.frombuffer(data, dtype=np.float32)
    loud_threshold = np.mean(np.abs(noise_sample)) * 10
    audio_buffer = []
    frames = []
    near = 0

    print("Start recording...")
    start = time.time()

    while((time.time() - start)  < 10):
        # Read chunk and load it into numpy array.
        data = stream.read(CHUNKSIZE)
        frames.append(data)
        print(time.time() - start)
        current_window = np.frombuffer(data, dtype=np.float32)

        # Reduce noise real-time
        current_window = nr.reduce_noise(
            audio_clip=current_window, noise_clip=noise_sample, verbose=False)

        if (len(audio_buffer) == 0):
            audio_buffer = current_window
        else:
            if(np.mean(np.abs(current_window)) < loud_threshold):
                # print("Inside silence reign")
                if(near < 1):
                    audio_buffer = np.concatenate(
                        (audio_buffer, current_window))
                    near += 1
                else:
                    predicted_audio = predict(model, np.array(audio_buffer))
                    print(predicted_audio)
                    audio_buffer = []
                    near
            else:
                print("Inside loud reign")
                near = 0
                audio_buffer = np.concatenate((audio_buffer, current_window))

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
