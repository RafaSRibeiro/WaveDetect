import os
import recognition
from pydub import AudioSegment
from pydub.utils import make_chunks

file_name = './test/clipe2.wav'

source_snare = AudioSegment.from_wav("./source/snare.wav")
source_kick = AudioSegment.from_wav("./source/kick.wav")

audio = AudioSegment.from_file(file_name, "wav")
audio = audio.set_frame_rate(16000)
audio = audio.set_sample_width(2)
audio = audio.split_to_mono()
audio = audio[0]
chunk_length_ms = 10  # pydub calculates in millisecond
chunks = make_chunks(audio, chunk_length_ms)  # Make chunks of one sec

threshold_trigger = False
temporary_raw_sample = []
current_time = 0
current_temporary_raw_sample_time = 0
# Convert chunks to raw audio data which you can then feed to HTTP stream
output_snare = AudioSegment(data=b''.join([]), sample_width=2, frame_rate=32000, channels=2)
output_snare_positions = []

output_kick = AudioSegment(data=b''.join([]), sample_width=2, frame_rate=32000, channels=2)
output_kick_positions = []

for i, chunk in enumerate(chunks):
    current_time += chunk_length_ms
    raw_audio_data = chunk.raw_data
    sound = AudioSegment(data=b''.join([raw_audio_data]), sample_width=2, frame_rate=16000, channels=1)
    if sound.max_dBFS > -6:
        threshold_trigger = True

    if threshold_trigger:
        if not temporary_raw_sample:
            current_temporary_raw_sample_time = current_time
        temporary_raw_sample.append(raw_audio_data)
        if len(temporary_raw_sample) >= 30:
            temporary_sound = AudioSegment(data=b''.join(temporary_raw_sample), sample_width=2, frame_rate=16000,
                                           channels=1)
            temp_filename = './temp.wav'
            temporary_sound.export(temp_filename, format='wav')
            prediction = recognition.predict(temp_filename)
            if prediction == 'snare':
                # print(prediction, current_temporary_raw_sample_time / 1000)
                # output_snare += AudioSegment.silent(chunk_length_ms)
                output_snare_positions.append(current_temporary_raw_sample_time)
            if prediction == 'kick':
                # print(prediction, current_temporary_raw_sample_time / 1000)
                # output_kick += source_kick
                output_kick_positions.append(current_temporary_raw_sample_time)
                # output_kick += AudioSegment.silent(chunk_length_ms)
            temporary_raw_sample = []
            threshold_trigger = False
    output_kick += AudioSegment.silent(chunk_length_ms)
    output_snare += AudioSegment.silent(chunk_length_ms)

for position in output_kick_positions:
    output_kick = output_kick.overlay(source_kick, position=position)

for position in output_snare_positions:
    output_snare = output_snare.overlay(source_snare, position=position)


output_snare.export('./output/snare.wav', format='wav')
output_kick.export('./output/kick.wav', format='wav')
print(output_snare_positions)
print(output_kick_positions)
