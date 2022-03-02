import soundfile as sf
import pyloudnorm as pyln
import wave

# data, rate = sf.read("./test/snare2.wav")  # load audio (with shape (samples, channels))
# print(rate)
# meter = pyln.Meter(rate)  # create BS.1770 meter
# loudness = meter.integrated_loudness(data)  # measure loudness
# print(loudness)
wr = wave.open('./test/snare2.wav', 'rb')
nchannels, sampwidth, framerate, nframes, comptype, compname = wr.getparams()
print(nchannels, sampwidth, framerate, nframes, comptype, compname)

wr = wave.open('./file.wav', 'rb')
nchannels, sampwidth, framerate, nframes, comptype, compname = wr.getparams()
print(nchannels, sampwidth, framerate, nframes, comptype, compname)