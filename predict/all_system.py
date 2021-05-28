import sys
sys.path.append("../")
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

from train.model import Confirming_model, Food_model
from predict_confirming import ConfirmingPrediction
from predict_food_number import FoodNumberPrediction

import datetime
import random
import wave
import pyaudio
import noisereduce as nr
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf

def id_generator():
    now = datetime.datetime.now()
    table_number = random.randint(0, 20)

    return now.strftime("%Y") + now.strftime("%m") + now.strftime("%d") + now.strftime("%H") + now.strftime("%M") + now.strftime("%S") + str(table_number)

SAVE_AUDIO_FILE_PATH = "../../recorded_audios/further_training/" + id_generator() + ".wav"
CONFIRMING_MODEL_PATH = "../train/model_confirming.h5"
FOOD_NUMBER_MODEL_PATH = "../train/model_food_number.h5"

# System audio path 
WELCOME_PATH = "../../recorded_audios/system_audio/welcome.wav"
ASK_ORDER_NTH_PATH = "../../recorded_audios/system_audio/ask_order_nth.wav"
ORDER_AGAIN_PATH = "../../recorded_audios/system_audio/order_again.wav"
ORDER_FAILURE_PATH = "../../recorded_audios/system_audio/order_failure.wav"
ORDER_MORE_PATH = "../../recorded_audios/system_audio/order_more.wav"
ORDER_SUCCESS_PATH = "../../recorded_audios/system_audio/order_success.wav"
NOT_UNDERSTAND_ORDER = "../../recorded_audios/system_audio/not_understand_order.wav"

CONFIRM_DISH_1ST_PATH = ["../../recorded_audios/system_audio/confirm_order_ca_kho_1st.wav", "../../recorded_audios/system_audio/confirm_order_ca_xot_1st.wav", \
    "../../recorded_audios/system_audio/confirm_order_khoai_tay_chien_1st.wav", "../../recorded_audios/system_audio/confirm_order_com_heo_xi_muoi_1st.wav", \
        "../../recorded_audios/system_audio/confirm_order_com_nieu_1st.wav", "../../recorded_audios/system_audio/confirm_order_com_tam_1st.wav", \
            "../../recorded_audios/system_audio/confirm_order_com_thap_cam_1st.wav", "../../recorded_audios/system_audio/confirm_order_rau_cai_luoc_1st.wav",\
                 "../../recorded_audios/system_audio/confirm_order_rau_cai_xao_1st.wav", "../../recorded_audios/system_audio/confirm_order_salad_tron_1st.wav",\
                     "../../recorded_audios/system_audio/confirm_order_tra_hoa_cuc_1st.wav", "../../recorded_audios/system_audio/confirm_order_tra_sam_dua_1st.wav", \
                         "../../recorded_audios/system_audio/confirm_order_trung_chien_1st.wav"]

CONFIRM_DISH_NTH_PATH = ["../../recorded_audios/system_audio/confirm_order_ca_kho_nth.wav", "../../recorded_audios/system_audio/confirm_order_ca_xot_nth.wav", \
    "../../recorded_audios/system_audio/confirm_order_khoai_tay_chien_nth.wav", "../../recorded_audios/system_audio/confirm_order_com_heo_xi_muoi_nth.wav", \
        "../../recorded_audios/system_audio/confirm_order_com_nieu_nth.wav", "../../recorded_audios/system_audio/confirm_order_com_tam_nth.wav", \
            "../../recorded_audios/system_audio/confirm_order_com_thap_cam_nth.wav", "../../recorded_audios/system_audio/confirm_order_rau_cai_luoc_nth.wav",\
                 "../../recorded_audios/system_audio/confirm_order_rau_cai_xao_nth.wav", "../../recorded_audios/system_audio/confirm_order_salad_tron_nth.wav",\
                     "../../recorded_audios/system_audio/confirm_order_tra_hoa_cuc_nth.wav", "../../recorded_audios/system_audio/confirm_order_tra_sam_dua_nth.wav", \
                         "../../recorded_audios/system_audio/confirm_order_trung_chien_nth.wav"]

CHUNKSIZE = 16000  # fixed chunk size
RATE = 16000
SAMPLE_FORMAT = pyaudio.paFloat32
CHANNELS = 1

# ERROR_HANDLER_FUNC = CFUNCTYPE(
#     None, c_char_p, c_int, c_char_p, c_int, c_char_p)

class SystemNotUnderstand(Exception):
    pass

# def py_error_handler(filename, line, function, err, fmt):
#     pass

class AllSystem:
    def __init__(self):
        super(AllSystem, self).__init__()
        self.SYSTEM_UNDERSTAND = True

        # Handle self.streaming error
        # c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        # asound = cdll.LoadLibrary('libasound.so')
        # asound.snd_lib_error_set_handler(c_error_handler)

        # initialize portaudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=SAMPLE_FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

    def system_say(self, audio_path):
        self.stream.stop_stream()
        data, fs = sf.read(audio_path, dtype='float32') 
        sd.play(data, fs)
        sd.wait()

    def user_reply(self, noise_sample, prediction, model,user_response_type):
        user_response_content = ""
        system_understand = False
        frame = []

        predicted_window = np.array([])

        times_trying_understand = 1

        while not system_understand:
            if self.stream.is_stopped():
                self.stream.start_stream()

            data = self.stream.read(CHUNKSIZE)
            frame.append(data)
            current_window = np.frombuffer(data, dtype=np.float32)

            if(np.amax(current_window) > 0.49):
                current_window = nr.reduce_noise(audio_clip=current_window, noise_clip=noise_sample, verbose=False)
                predicted_window = np.append(predicted_window, current_window)
            else:
                if(len(predicted_window) == 0):
                    noise_sample = np.frombuffer(data, dtype=np.float32)
                else:
                    user_response_content = prediction.predict(model, np.array(predicted_window))
                    print(user_response_content)
                    predicted_window = np.array([])


                    # Not understand solution
                    system_understand = self.system_understand_f(user_response_content, user_response_type)

                    if not system_understand:
                        if times_trying_understand < 3:
                            self.system_say(NOT_UNDERSTAND_ORDER)
                            times_trying_understand += 1
                        else:
                            return "", noise_sample, frame
                    else:
                        return user_response_content, noise_sample, frame

    def system_understand_f(self, user_response_content, user_response_type):
        confirming_labels = ["co", "khong"]
        food_number_labels = ["ca_kho", "ca_xot", "khoai_tay_chien", "com_heo_xi_muoi", "com_nieu", "com_tam", \
            "com_thap_cam", "rau_cai_luoc", "rau_cai_xao", "salad_tron", "tra_hoa_cuc", "tra_sam_dua", "trung_chien"]

        if(user_response_type == "confirming"):
            if(user_response_content in confirming_labels):
                return True
            else:
                self.SYSTEM_UNDERSTAND = False
                return False

        if(user_response_type == "food_number"):
            if(user_response_content in food_number_labels):
                return True
            else:
                self.SYSTEM_UNDERSTAND = False
                return False
        else:
            return None

    def find_confirmed_dish_number_path(self, user_response, time):
        food_number_labels = ["ca_kho", "ca_xot", "khoai_tay_chien", "com_heo_xi_muoi", "com_nieu", "com_tam", \
            "com_thap_cam", "rau_cai_luoc", "rau_cai_xao", "salad_tron", "tra_hoa_cuc", "tra_sam_dua", "trung_chien"]

        if time == 1:
            return CONFIRM_DISH_1ST_PATH[food_number_labels.index(user_response)]
        else:
            return CONFIRM_DISH_NTH_PATH[food_number_labels.index(user_response)]

    def listToString(self, s): 
    
        # initialize an empty string
        str1 = ", " 
        
        # return string  
        return (str1.join(s))
        
    
    def start(self):
        device = torch.device("cpu")

        # Confirming model initialization
        confirming_model = Confirming_model(Confirming_model.hparams['n_mels'], Confirming_model.hparams['cnn_channels'], Confirming_model.hparams['cnn_kernel_size'], \
            Confirming_model.hparams['stride'], Confirming_model.hparams['gru_hidden_size'], Confirming_model.hparams['attention_hidden_size'], Confirming_model.hparams['n_classes']).to(device)

        confirming_model_checkpoint = torch.load(CONFIRMING_MODEL_PATH, map_location=device)
        confirming_model.load_state_dict(confirming_model_checkpoint)
        confirming_model.eval()

        confirming_prediction = ConfirmingPrediction()

        # Food model initialization 
        food_number_model = Food_model(Food_model.hparams['n_mels'], Food_model.hparams['cnn_channels'], Food_model.hparams['cnn_kernel_size'], \
            Food_model.hparams['stride'], Food_model.hparams['gru_hidden_size'], Food_model.hparams['attention_hidden_size'], Food_model.hparams['n_classes']).to(device)

        food_number_checkpoint = torch.load(FOOD_NUMBER_MODEL_PATH, map_location=device)
        food_number_model.load_state_dict(food_number_checkpoint)
        food_number_model.eval()

        food_number_prediction = FoodNumberPrediction()

        # noise window
        data = self.stream.read(CHUNKSIZE)
        noise_sample = np.frombuffer(data, dtype=np.float32)
        # loud_threshold = np.mean(np.abs(noise_sample)) * 10
        all_frames = []

        try:
            # System welcome customers
            start_conversation = True
            order_more = False
            all_dishes_ordered = []
            order_fail = False

            while order_more or start_conversation:
                if start_conversation:
                    self.system_say(WELCOME_PATH)
                else:
                    self.system_say(ASK_ORDER_NTH_PATH)

                time_order_fail_successively = 0
                one_order_sucess = False
                
                while(time_order_fail_successively < 3) and one_order_sucess == False:
                    if (time_order_fail_successively != 0):
                        self.system_say(ORDER_AGAIN_PATH)

                    user_response, noise_sample, frame = self.user_reply(noise_sample, food_number_prediction, food_number_model,"food_number")
                    
                    all_dishes_ordered.append(user_response)

                    for each_frame in frame:
                        all_frames.append(each_frame)
                    
                    if start_conversation:
                        self.system_say(self.find_confirmed_dish_number_path(user_response, 1))
                        start_conversation = False
                    else:
                        self.system_say(self.find_confirmed_dish_number_path(user_response, 2))

                    user_response, noise_sample, frame = self.user_reply(noise_sample, confirming_prediction, confirming_model,"confirming")

                    for each_frame in frame:
                        all_frames.append(each_frame)

                    if (user_response == "khong"):
                        all_dishes_ordered.pop()
                        self.SYSTEM_UNDERSTAND = False
                        time_order_fail_successively += 1
                    if (user_response == "co"):
                        one_order_sucess = True

                if time_order_fail_successively == 3:
                    raise SystemNotUnderstand

                self.system_say(ORDER_MORE_PATH)

                user_response, noise_sample, frame = self.user_reply(noise_sample, confirming_prediction, confirming_model, "confirming")
                for each_frame in frame:
                    all_frames.append(each_frame)

                if (user_response == ""):
                    raise SystemNotUnderstand

                order_more = user_response == "co"
            
            print(all_dishes_ordered)
            self.system_say(ORDER_SUCCESS_PATH)

        except SystemNotUnderstand:
            order_fail = True
            self.system_say(ORDER_FAILURE_PATH)
            data = self.stream.read(CHUNKSIZE)
        finally:
            # close self.stream
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

            if not self.SYSTEM_UNDERSTAND:
                # Save the recorded data as a WAV file when not understanding appears in the conversation
                wf = wave.open(SAVE_AUDIO_FILE_PATH, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(SAMPLE_FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(all_frames))
                wf.close()

            if not order_fail:
                return self.listToString(all_dishes_ordered)
            else:
                return "Nhan vien order"

if __name__ == "__main__":
    allSystem = AllSystem()
    allSystem.start()