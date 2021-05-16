import sys
sys.path.append("../")

import os
import librosa

file_name = 'D:/Study/uni/DoAn/food_ordering_system/predict/test/four.wav'
file_noise = "D:/Study/uni/DoAn/food_ordering_system/data/restaurant_noise/03noise_restaurant04noise_restaurant.wav"

signal_name, sample_rate = librosa.load(file_name)
signal_noise, sample_rate = librosa.load(file_noise)
signal = []

if (len(signal_name) > len(signal_noise)):
    for i in range(len(signal_noise)):
        signal.append(signal_name[i] + signal_noise[i])
    for i in range(len(signal_name) - len(signal_noise)):
else:
    for i in range(len(signal_name)):
        signal.append(signal_name[i] + signal_noise[i])