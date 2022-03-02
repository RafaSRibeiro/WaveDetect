#!usr/bin/env python
# coding=utf-8

import pyaudio
from pydub.utils import mediainfo
import wave
import recognition


def save_wave(frames):
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(1)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(16000)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


# audio_files = ["./test/snare.wav", "file.wav"]
# open a wav format music
audio_file = wave.open(r"./test/snare.wav", "rb")
# audio_file = wave.open(r"file.wav", "rb")
# instantiate PyAudio
audio = pyaudio.PyAudio()
# open stream
# print('Samples', audio.get_format_from_width(audio_file.getsampwidth()))
# print('Channels', audio_file.getnchannels())
# print('Framerate', audio_file.getframerate())
stream = audio.open(
    format=audio.get_format_from_width(audio_file.getsampwidth()),
    channels=audio_file.getnchannels(),
    rate=audio_file.getframerate(),
    output=True
)

FORMAT = audio.get_format_from_width(audio_file.getsampwidth())
CHUNK = 1600
WAVE_OUTPUT_FILENAME = "./file.wav"
SAMPLE_RATE = audio_file.getframerate()

# read data
data = audio_file.readframes(CHUNK)

frames = []
# play stream
while data:
    stream.write(data)
    frames.append(data)
    save_wave(frames)
    recognition.predict(WAVE_OUTPUT_FILENAME)
    data = audio_file.readframes(CHUNK)

# stop stream
stream.stop_stream()
stream.close()

# close PyAudio
audio.terminate()
