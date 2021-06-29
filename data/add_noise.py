# This function is for adding noise to original dataset. 
# How much SNR to set can be found in this article 
# https://arxiv.org/abs/1703.05390. However, in the 
# project, I've not used this method for generating 
# data as it lower the precision when tested in real time

from scipy.io import wavfile
from scipy.io.wavfile import write
import numpy as np
import random
import os


random.seed(0)
SNR = 5

# DATASET_PATH = ["../../confirming_dataset/co/", "../../confirming_dataset/khong/", "../../confirming_dataset/khong_biet/"]

DATASET_PATH = ["../../food_dataset/ca_kho", "../../food_dataset/ca_xot", "../../food_dataset/com_ga", "../../food_dataset/com_heo_xi_muoi", "../../food_dataset/com_nieu", \
                    "../../food_dataset/com_tam", "../../food_dataset/com_thap_cam", "../../food_dataset/khong_biet", "../../food_dataset/rau_muong_luoc", \
                        "../../food_dataset/rau_muong_xao",  "../../food_dataset/salad_tron", "../../food_dataset/tra_hoa_cuc", "../../food_dataset/tra_sam_dua", \
                            "../../food_dataset/trung_chien"]

NOISE_PATH = ["/home/minhair/Desktop/food_ordering_system/food_ordering_system/predict/recorded_audios/environment_noise/medium_without_subject_voice", 
        "/home/minhair/Desktop/food_ordering_system/food_ordering_system/predict/recorded_audios/environment_noise/low_without_subject_voice"]

# SAVED_NEW_PATH = ["../../confirming_with_noise_dataset/co/", "../../confirming_with_noise_dataset/khong/", "../../confirming_with_noise_dataset/khong_biet/"]

SAVED_NEW_PATH = ["../../food_with_noise_dataset/ca_kho/", "../../food_with_noise_dataset/ca_xot/", "../../food_with_noise_dataset/com_ga/", "../../food_with_noise_dataset/com_heo_xi_muoi/", "../../food_with_noise_dataset/com_nieu/", \
                    "../../food_with_noise_dataset/com_tam/", "../../food_with_noise_dataset/com_thap_cam/", "../../food_with_noise_dataset/khong_biet/", "../../food_with_noise_dataset/rau_muong_luoc/", \
                        "../../food_with_noise_dataset/rau_muong_xao/",  "../../food_with_noise_dataset/salad_tron/", "../../food_with_noise_dataset/tra_hoa_cuc/", "../../food_with_noise_dataset/tra_sam_dua/", \
                            "../../food_with_noise_dataset/trung_chien/"]


def add_noise(dataset_path, noise_path, saved_file_path):
    for index, (data_set, saved_set) in enumerate(zip(dataset_path, saved_file_path)):
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_set)):
            for file_name_original in filenames:
            # Load the original file
                file_path = os.path.join(dirpath, file_name_original)
                fs, data = wavfile.read(file_path)
                data = data.astype(np.float32)

                for noise_set in noise_path:
                    #There's low noise and medium noise added. The ratio of low noise and medium noise is taking is 2/3
                    if "medium_without_subject_voice" in noise_set:
                        for k, (dirpath_noise, dirnames_noise, filenames_noise) in enumerate(os.walk(noise_set)):
                            for j in range(1, 16):
                                # randomly get the noise file
                                file_name_noise = random.choice(filenames_noise)
                                example_noise_file_path = os.path.join(dirpath_noise, file_name_noise)

                                # Loading noise file
                                _, noise = wavfile.read(example_noise_file_path)
                                noise = noise.astype(np.float32)

                                #Adding noise
                                while noise.shape[0] < data.shape[0]:
                                    noise = np.concatenate((noise, noise), axis=0)
                                offset = random.randint(0, noise.shape[0]-data.shape[0])
                                noise = noise[offset:offset+data.shape[0]]

                                data_power = np.dot(data, data) / data.shape[0]
                                noise_power = np.dot(noise, noise) / noise.shape[0]
                                scale_factor = np.sqrt(np.power(10, -SNR / 10) * data_power / noise_power)

                                new_data = data + noise * scale_factor

                                wavfile.write(saved_set + file_name_original + str(j) + ".wav", fs, new_data.astype(np.float32))

                    else:
                        for k, (dirpath_noise, dirnames_noise, filenames_noise) in enumerate(os.walk(noise_set)):
                            for j in range(16, 26):
                                file_name_noise = random.choice(filenames_noise)
                                example_noise_file_path = os.path.join(dirpath_noise, file_name_noise)

                                _, noise = wavfile.read(example_noise_file_path)
                                noise = noise.astype(np.float32)

                                while noise.shape[0] < data.shape[0]:
                                    noise = np.concatenate((noise, noise), axis=0)
                                offset = random.randint(0, noise.shape[0]-data.shape[0])
                                noise = noise[offset:offset+data.shape[0]]

                                data_power = np.dot(data, data) / data.shape[0]
                                noise_power = np.dot(noise, noise) / noise.shape[0]
                                scale_factor = np.sqrt(np.power(10, -SNR / 10) * data_power / noise_power)

                                new_data = data + noise * scale_factor

                                wavfile.write(saved_set + file_name_original + str(j) + ".wav", fs, new_data.astype(np.float32))


if __name__ == "__main__":
    add_noise(DATASET_PATH, NOISE_PATH, SAVED_NEW_PATH)
                