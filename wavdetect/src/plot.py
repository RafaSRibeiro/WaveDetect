import matplotlib.pyplot as plt
import numpy as np
import wave
import sys


# Extract Raw Audio from Wav File
def get_wavform(file_path):
    spf = wave.open(file_path, "r")
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, int)

    # If Stereo
    if spf.getnchannels() == 2:
        print("Just mono files")
        sys.exit(0)

    return signal
    # plt.figure(1)
    # plt.title("Signal Wave...")
    # plt.plot(signal)
    # plt.show()
