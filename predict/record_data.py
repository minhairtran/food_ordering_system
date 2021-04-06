import sounddevice
from scipy.io.wavfile import write
import string
import random

SAMPLE_RATE = 22050
CHANNELS = 2
SECOND = 3


def id_generator(size=17, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

if __name__ == "__main__":
    """
    for i in ("yes", "no"):
        print("recording......")
        recorded_voice = sounddevice.rec(int(SECOND * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
        sounddevice.wait()
        write("/home/minhair/Desktop/food_ordering_system/confirming_dataset/" + str(i) + "/" + id_generator() + ".wav", SAMPLE_RATE, recorded_voice)
    """

    for i in range(8, 9):
        print("recording......")
        recorded_voice = sounddevice.rec(int(SECOND * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
        sounddevice.wait()
        write("/home/minhair/Desktop/test_pytorch/food_ordering_system/predict/test/no" + str(i) + ".wav", SAMPLE_RATE, recorded_voice)