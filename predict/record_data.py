# This is for recording audio (noise or keyword)

import sounddevice
from scipy.io.wavfile import write
import string
import random

SAMPLE_RATE = 16000
CHANNELS = 1
#Second recorded
SECOND = 3


def id_generator(random_number, size=2, chars=string.ascii_uppercase + string.digits):
    return ''.join(random_number + "noise_restaurant")

if __name__ == "__main__":
    """
    for i in ("yes", "no"):
        print("recording......")
        recorded_voice = sounddevice.rec(int(SECOND * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
        sounddevice.wait()
        write("/home/minhair/Desktop/food_ordering_system/confirming_dataset/" + str(i) + "/" + id_generator() + ".wav", SAMPLE_RATE, recorded_voice)
    """

    # for i in range(20, 27):
    print("recording......")
    recorded_voice = sounddevice.rec(int(SECOND * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sounddevice.wait()
    # write("/home/minhair/Desktop/food_ordering_system/food_ordering_system/data/restaurant_noise/" + id_generator(str(i)) +  ".wav", SAMPLE_RATE, recorded_voice)
    write("C:/Users/minHair/OneDriveHanoiUniversityofScienceandTechnology/Desktop/confirm_order_khoai_tay_chien_nth.wav", SAMPLE_RATE, recorded_voice)
    sounddevice.sleep(1)

    # write("/home/minhair/Desktop/food_ordering_system/food_ordering_system/predict/recorded_audios/system_audio/" ".wav", SAMPLE_RATE, recorded_voice)