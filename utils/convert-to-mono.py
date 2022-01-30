# Python3 program to illustrate
# splitting stereo audio to mono
# using pydub

# Import AudioSegment from pydub
from pydub import AudioSegment
import os

base_dir = './work/dataset/learning/'

paths = os.listdir(base_dir)

for path in paths:
    export_path = os.path.join("./work/dataset/mono", path)
    os.makedirs(export_path, exist_ok=True)
    files = os.listdir(base_dir + path)
    for file in files:
        audio = base_dir + path + "/" + file
        stereo_audio = AudioSegment.from_file(audio, format="wav")
        mono_audios = stereo_audio.split_to_mono()
        export_file = os.path.join(export_path, file)
        mono_left = mono_audios[0].export(export_file.replace(" ", ""), format="wav")
