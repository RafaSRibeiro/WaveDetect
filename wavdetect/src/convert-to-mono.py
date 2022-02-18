# Python3 program to illustrate
# splitting stereo audio to mono
# using pydub

# Import AudioSegment from pydub
from pydub import AudioSegment
import os

base_dir = './work/dataset/learning/'

paths = os.listdir(base_dir)


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


for path in paths:
    export_mono_path = os.path.join("./work/dataset/mono", path)
    os.makedirs(export_mono_path, exist_ok=True)

    export_normalized_path = os.path.join("./work/dataset/normalized", path)
    os.makedirs(export_normalized_path, exist_ok=True)

    files = os.listdir(base_dir + path)
    for file in files:
        audio = base_dir + path + "/" + file
        print(audio)
        stereo_audio = AudioSegment.from_file(audio, format="wav")
        mono_audios = stereo_audio.split_to_mono()
        export_mono_file = os.path.join(export_mono_path, file).replace(" ", "")
        mono_left = mono_audios[0].export(export_mono_file, format="wav")
        sound = AudioSegment.from_file(export_mono_file, "wav")
        export_normalized_file = os.path.join(export_normalized_path, file).replace(" ", "")
        normalized_sound = match_target_amplitude(sound, 0.0)
        normalized_sound.export(export_normalized_file, format="wav")
