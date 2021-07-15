# import shutil
# import os

# folder = "C:/Users/minHair/OneDriveHanoiUniversityofScienceandTechnology/Desktop/hai-tieng-anh/data/fpt/khong"
# target = "C:/Users/minHair/OneDriveHanoiUniversityofScienceandTechnology/Desktop/food_dataset/khong_biet"


# for i, (dirpath, dirnames, filenames) in enumerate(os.walk(folder)):
#     for f in filenames:
#         for j in range(1521, 1637, 2):
#             if f.startswith(str(j)):
#                 print(f)
#                 shutil.copy2(folder + "/"+ f, target + "/khong" + f) 

import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt

scale_file = "test.wav"

scale, sr = librosa.load(scale_file)

mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=40)

log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)


plt.figure(figsize=(15, 10))
librosa.display.specshow(log_mel_spectrogram, 
                         x_axis="time",
                         y_axis="mel", 
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()