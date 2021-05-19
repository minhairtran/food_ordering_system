from scipy.io import wavfile
from scipy.io.wavfile import write
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(0)
SNR = 5

fs, data = wavfile.read("/home/minhair/Desktop/food_ordering_system/food_ordering_system/train/hai.wav")
data = data.astype(np.float32)

_, noise = wavfile.read("/home/minhair/Desktop/food_ordering_system/food_ordering_system/train/202105191927138.wav")
noise = noise.astype(np.float32)

while noise.shape[0] < data.shape[0]:
    noise = np.concatenate((noise, noise), axis=0)
offset = random.randint(0, noise.shape[0]-data.shape[0])
noise = noise[offset:offset+data.shape[0]]

data_power = np.dot(data, data) / data.shape[0]
noise_power = np.dot(noise, noise) / noise.shape[0]
scale_factor = np.sqrt(np.power(10, -SNR / 10) * data_power / noise_power)

new_data = data + noise * scale_factor

wavfile.write("/home/minhair/Desktop/food_ordering_system/food_ordering_system/train/new.wav", fs, new_data.astype(np.float32))