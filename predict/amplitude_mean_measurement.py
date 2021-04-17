import matplotlib.pyplot as plt
import librosa
import numpy as np
import os
import pandas as pd
from openpyxl import load_workbook
import pyaudio
import noisereduce as nr
from ctypes import *
import datetime
import random
import wave
import time

CHUNKSIZE = 22050  # fixed chunk size
RATE = 22050
SAMPLE_FORMAT = pyaudio.paFloat32
CHANNELS = 2


ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)


def py_error_handler(filename, line, function, err, fmt):
    pass

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

book = load_workbook('/home/minhair/Desktop/food_ordering_system/food_ordering_system/predict/recorded_audios/report_photos/report.xlsx')
writer = pd.ExcelWriter('/home/minhair/Desktop/food_ordering_system/food_ordering_system/predict/recorded_audios/report_photos/report.xlsx', engine = 'openpyxl')
writer.book = book

# Voice mean amplitude measurement
# mean_amplitude = {'mean_amplitude': [], 'file_name': []}

# for i, (dirpath, dirnames, filenames) in enumerate(os.walk("/home/minhair/Desktop/food_ordering_system/food_ordering_system/predict/test/")):
#     for f in filenames:
#         file_path = os.path.join(dirpath, f)
#         signal, sample_rate = librosa.load(file_path)
#         signal = nr.reduce_noise(audio_clip=signal, noise_clip=noise_sample, verbose=False)
#         mean_amplitude['mean_amplitude'].append(np.mean(np.abs(signal)))
#         mean_amplitude['file_name'].append(f)

# df1 = pd.DataFrame(mean_amplitude, columns = ['mean_amplitude', 'file_name'])

# df1.to_excel (writer, sheet_name='voice_mean_amplitude', index = False, header=True)

# Environment noise mean amplitude measurement
def id_generator():
    now = datetime.datetime.now()
    table_number = random.randint(0, 20)

    return now.strftime("%Y") + now.strftime("%m") + now.strftime("%d") + now.strftime("%H") + now.strftime("%M") + now.strftime("%S") + str(table_number)

FILENAME = "/home/minhair/Desktop/food_ordering_system/food_ordering_system/predict/recorded_audios/" + id_generator() + ".wav", 'wb'

mean_amplitude = {'time_(s)': [], 'mean_amplitude': [], 'file_name': []}
frames = []

print("Start recording...")
start = time.time()

while(time.time() - start < 60):
    data = stream.read(CHUNKSIZE)
    frames.append(data)
    current_window = np.frombuffer(data, dtype=np.float32)

    current_window = nr.reduce_noise(audio_clip=current_window, noise_clip=noise_sample, verbose=False)

    print((time.time() - start), np.mean(np.abs(current_window)))

    mean_amplitude['time_(s)'].append(time.time() - start)
    mean_amplitude['mean_amplitude'].append(np.mean(np.abs(current_window)))
    mean_amplitude['file_name'].append(FILENAME)

# close stream
stream.stop_stream()
stream.close()
p.terminate()

print('Finished recording')

# Save the recorded data as a WAV file
wf = wave.open("/home/minhair/Desktop/food_ordering_system/food_ordering_system/predict/recorded_audios/" + id_generator() + ".wav", 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

df2 = pd.DataFrame(mean_amplitude, columns = ['time_(s)', 'mean_amplitude', 'file_name'])

df2.to_excel (writer, sheet_name='low_noise_environment', startrow=12, index = False, header=False, columns = ['time_(s)', 'mean_amplitude', 'file_name'])

writer.save()
writer.close()


