import sounddevice
from scipy.io.wavfile import write
import string
import random

SAMPLE_RATE = 22050
CHANNELS = 2
SECOND = 5


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

    print("recording......")
    recorded_voice = sounddevice.rec(int(SECOND * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sounddevice.wait()
    write("/home/minhair/Desktop/food_ordering_system/food_ordering_system/predict/test/ask_order_nth.wav", SAMPLE_RATE, recorded_voice)