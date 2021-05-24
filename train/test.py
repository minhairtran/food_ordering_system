import soundfile as sf
import sounddevice as sd
# https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav
data, fs = sf.read('D:/Study/uni/DoAn/food_ordering_system/train/1409_leminh_nam_bac.wav')
print(data.shape,fs)
