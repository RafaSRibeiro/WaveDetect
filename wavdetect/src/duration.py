import wave
import contextlib

fname = 'file.wav'


def getWaveDuration():
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        print(duration)


def getWaveInfo():
    w = wave.open('./test/snare.wav', 'rb')
    print("Number of channels is: ", w.getnchannels())
    print("Sample width in bytes is: ", w.getsampwidth())
    print("Framerate is: ", w.getframerate())
    print("Number of frames is: ", w.getnframes())


if __name__ == "__main__":
    getWaveInfo()
