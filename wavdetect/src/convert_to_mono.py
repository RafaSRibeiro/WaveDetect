import os
from pydub import AudioSegment
from remove_silence import remove_initial_silence

base_dir = '../../jupyter/work/dataset/learning/'

paths = os.listdir(base_dir)

audio_length_ms = 100

for path in paths:
    export_mono_path = os.path.join("./mono"+str(audio_length_ms)+"ms", path)
    os.makedirs(export_mono_path, exist_ok=True)

    files = os.listdir(base_dir + path)
    for index, file in enumerate(files, start=1):
        audio_file = base_dir + path + "/" + file
        audio = AudioSegment.from_wav(audio_file)
        audio = remove_initial_silence(audio)
        if len(audio) > audio_length_ms:
            audio = audio[0:audio_length_ms]
            audio = audio.set_frame_rate(16000)
            audios = audio.split_to_mono()
            export_mono_file = os.path.join(export_mono_path, path + str(index) + '.wav').replace(" ", "")
            mono_left = audios[0].export(export_mono_file, format="wav")
            sound = AudioSegment.from_file(export_mono_file, "wav")
