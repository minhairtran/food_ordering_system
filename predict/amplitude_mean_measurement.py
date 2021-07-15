# Recording environment noise 
import numpy as np
import pandas as pd
import pyaudio
from ctypes import *
import datetime
import random
import wave
import time

CHUNKSIZE = 16000  # fixed chunk size
RATE = 16000
SAMPLE_FORMAT = pyaudio.paFloat32
CHANNELS = 1


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

# Init result excel file
writer = pd.ExcelWriter('/home/minhair/Desktop/food_ordering_system/food_ordering_system/predict/recorded_audios/environment_noise/report.xlsx', engine = 'openpyxl')

# Environment noise mean amplitude measurement
def id_generator():
    now = datetime.datetime.now()
    random_num = random.randint(0, 20)

    return now.strftime("%Y") + now.strftime("%m") + now.strftime("%d") + now.strftime("%H") + now.strftime("%M") + now.strftime("%S") + str(random_num)

# Time; max local amplitude and file name are saved 
mean_amplitude = {'time_(s)': [], 'max_local_amplitude': [], 'file_name': []}
frames = []

print("Start recording...")
start = time.time()

# 303 points made
while(time.time() - start < 303):
    frames = []
    data = stream.read(CHUNKSIZE)
    frames.append(data)
    current_window = np.frombuffer(data, dtype=np.float32)

    print((time.time() - start), np.amax(current_window))

    FILENAME = "/home/minhair/Desktop/food_ordering_system/food_ordering_system/predict/recorded_audios/environment_noise/low_without_subject_noise/" + id_generator() + ".wav"

    mean_amplitude['time_(s)'].append(time.time() - start)
    mean_amplitude['max_local_amplitude'].append(np.amax(current_window))
    mean_amplitude['file_name'].append(FILENAME)

    # Save the recorded data as a WAV file
    wf = wave.open(FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    time.sleep(1)

# close stream
stream.stop_stream()
stream.close()
p.terminate()

print('Finished recording')

df2 = pd.DataFrame(mean_amplitude, columns = ['time_(s)', 'max_local_amplitude', 'file_name'])

df2.to_excel (writer, sheet_name='voice_live_environment', startrow=1, index = False, header=True, columns = ['time_(s)', 'max_local_amplitude', 'file_name'])

writer.save()
writer.close()


