import sys
sys.path.append("../")
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

from train.model import Confirming_model, Food_model
from predict_confirming import ConfirmingPrediction
from predict_food_number import FoodNumberPrediction

import datetime
import pandas as pd
import wave
import pyaudio
import torch
import numpy as np
import noisereduce as nr

def id_generator(gender, region, masking):
    now = datetime.datetime.now()
    return str(gender) + "_" + str(region) + "_" + str(masking) + "_" + now.strftime("%Y") + now.strftime("%m") + now.strftime("%d")

SAVE_AUDIO_FILE_PATH = "../../recorded_audios/final_experiment/"
CONFIRMING_MODEL_PATH = "../train/model_confirming_"
FOOD_NUMBER_MODEL_PATH = "../train/model_food_number_"

CHUNKSIZE = 16000  # fixed chunk size
RATE = 16000
SAMPLE_FORMAT = pyaudio.paFloat32
CHANNELS = 1

def keyword_said(keyword_number):
    all_keyword = {
        0: "co",
        1: "khong",
        2: "com_tam",
        3: "com_nieu",
        4: "khoai_tay_chien",
        5: "com_thap_cam",
        6: "com_heo_xi_muoi",
        7: "ca_kho",
        8: "ca_xot",
        9: "trung_chien",
        10: "rau_cai_luoc",
        11: "rau_cai_xao",
        12: "salad_tron",
        13: "tra_sam_dua",
        14: "tra_hoa_cuc",
        15: "khong_biet",
    }

    return all_keyword[keyword_number]

def main(masking, save_audio_file_path):
    device = torch.device("cpu")
    confirming_model_path = CONFIRMING_MODEL_PATH + str(masking) + ".h5"

    # Confirming model initialization
    confirming_model = Confirming_model(Confirming_model.hparams['n_mels'], Confirming_model.hparams['cnn_channels'], Confirming_model.hparams['cnn_kernel_size'], \
        Confirming_model.hparams['stride'], Confirming_model.hparams['gru_hidden_size'], Confirming_model.hparams['attention_hidden_size'], Confirming_model.hparams['n_classes']).to(device)

    confirming_model_checkpoint = torch.load(confirming_model_path, map_location=device)
    confirming_model.load_state_dict(confirming_model_checkpoint)
    confirming_model.eval()

    confirming_prediction = ConfirmingPrediction()

    # Food model initialization 
    food_number_model_path = FOOD_NUMBER_MODEL_PATH + str(masking) + ".h5"

    food_number_model = Food_model(Food_model.hparams['n_mels'], Food_model.hparams['cnn_channels'], Food_model.hparams['cnn_kernel_size'], \
        Food_model.hparams['stride'], Food_model.hparams['gru_hidden_size'], Food_model.hparams['attention_hidden_size'], Food_model.hparams['n_classes']).to(device)

    food_number_checkpoint = torch.load(food_number_model_path, map_location=device)
    food_number_model.load_state_dict(food_number_checkpoint)
    food_number_model.eval()

    food_number_prediction = FoodNumberPrediction()

    # excel
    writer = pd.ExcelWriter('result.xlsx', engine = 'openpyxl')

    result = {'keyword_said': [], 'result': [], 'file_recorded_name': ''}

    # streaming
    p = pyaudio.PyAudio()
    stream = p.open(format=SAMPLE_FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

    data = stream.read(CHUNKSIZE)
    noise_sample = np.frombuffer(data, dtype=np.float32)
    frames = []
    predicted_window = np.array([])
    print("Start recording...")

    keyword_number = 0

    while(keyword_number < 16):
        # Read chunk and load it into numpy array.
        data = stream.read(CHUNKSIZE)
        frames.append(data)
        current_window = np.frombuffer(data, dtype=np.float32)

        if(np.amax(current_window) > 0.49):
            
            predicted_window = np.append(predicted_window, current_window)
        else:
            if(len(predicted_window) == 0):
                current_window = nr.reduce_noise(audio_clip=current_window, noise_clip=noise_sample, verbose=False)
                # noise_sample = np.frombuffer(data, dtype=np.float32)
            else:
                result['keyword_said'].append(keyword_said(keyword_number))

                if (keyword_number < 2):
                    predicted_audio = confirming_prediction.predict(confirming_model, np.array(predicted_window))
                else:
                    predicted_audio = food_number_prediction.predict(food_number_model, np.array(predicted_window))
                if (predicted_audio == keyword_said(keyword_number)):
                    result['result'].append(1)
                else:
                    result['result'].append(0)     
                print(predicted_audio)
                predicted_window = np.array([])
                keyword_number += 1

    #terminate streaming
    stream.stop_stream()
    stream.close()
    p.terminate()

    print('Finished recording')

    # save record
    wf = wave.open(save_audio_file_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    #save result in excel
    result['file_recorded_name'] = save_audio_file_path
    df2 = pd.DataFrame(result, columns = ['keyword_said', 'result', 'file_recorded_name'])
    df2.to_excel (writer, sheet_name='result', startrow=0, index = False, header=True, columns = ['keyword_said', 'result', 'file_recorded_name'])
    writer.save()
    writer.close()

if __name__ == "__main__":
    frequency_time_masking = "13_5"
    save_audio_file_path = SAVE_AUDIO_FILE_PATH + id_generator("nam", "bac", frequency_time_masking) + ".wav"
    main(frequency_time_masking, save_audio_file_path)

