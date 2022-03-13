# Python3 program to illustrate
# splitting stereo audio to mono
# using pydub

# Import AudioSegment from pydub
from pydub import AudioSegment
from pydub.utils import make_chunks
import os

base_dir = '../../jupyter/work/dataset/learning/'

paths = os.listdir(base_dir)


def match_target_amplitude(audio, target_dBFS):
    change_in_dBFS = target_dBFS - audio.max_dBFS
    return audio.apply_gain(change_in_dBFS)


def remove_initial_silence(audio, threshold=-12):
    threshold_limit = False
    temporary_raw_sample = []
    chunk_length_ms = 10  # pydub calculates in millisecond
    chunks = make_chunks(audio, chunk_length_ms)  # Make chunks of one sec
    for i, chunk in enumerate(chunks):
        raw_data = chunk.raw_data
        if not threshold_limit:
            chunk = AudioSegment(data=b''.join([raw_data]), sample_width=2, frame_rate=16000, channels=1)
            if chunk.max_dBFS > threshold:
                threshold_limit = True
        else:
            temporary_raw_sample.append(raw_data)

    return AudioSegment(data=b''.join(temporary_raw_sample), sample_width=2, frame_rate=16000, channels=1)


for path in paths:
    export_mono_split_path = os.path.join("./work/dataset/mono_split", path)
    os.makedirs(export_mono_split_path, exist_ok=True)

    export_mono_path = os.path.join("./work/dataset/mono", path)
    os.makedirs(export_mono_path, exist_ok=True)

    export_normalized_path = os.path.join("./work/dataset/normalized", path)
    os.makedirs(export_normalized_path, exist_ok=True)

    files = os.listdir(base_dir + path)
    for file in files:
        audio_file = base_dir + path + "/" + file
        split_audio = AudioSegment.from_wav(audio_file)
        split_audio = split_audio[0:200]
        split_audio_file = export_mono_split_path + "/" + file
        # split_audio = remove_initial_silence(split_audio)
        split_audio.export(split_audio_file, format="wav")

        stereo_audio = AudioSegment.from_file(split_audio_file, format="wav")
        stereo_audio = stereo_audio.set_frame_rate(16000)
        mono_audios = stereo_audio.split_to_mono()
        export_mono_file = os.path.join(export_mono_path, file).replace(" ", "")
        mono_left = mono_audios[0].export(export_mono_file, format="wav")
        sound = AudioSegment.from_file(export_mono_file, "wav")
        export_normalized_file = os.path.join(export_normalized_path, file).replace(" ", "")
        normalized_sound = match_target_amplitude(sound, 0.0)
        normalized_sound.export(export_normalized_file, format="wav")

    # for file_name in files:
    #     file_path = os.path.join(base_dir + path, file_name)
    #     audio_segment = AudioSegment.from_file(file_path, format="wav")
    #     audio_segment = audio_segment.set_frame_rate(16000)
    #     audio_segment = audio_segment.set_sample_width(2)
    #     audio_segment = audio_segment.split_to_mono()
    #     audio_segment = remove_initial_silence(audio_segment[0])
    #     audio = audio_segment[:300]
    #     export_mono_file = os.path.join(export_mono_path, file_name).replace(" ", "")
    #     audio.export(export_mono_file, format='wav')