import os
import argparse
from recognition import Recognition
from pydub import AudioSegment
from pydub.utils import make_chunks
import normalize

parser = argparse.ArgumentParser(description="Drum Detect", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--chunk-length-ms", help="chunk length in ms", default=10)
parser.add_argument("-t", "--threshold", help="threshold in db", default=-6)
parser.add_argument("src", help="Input audio file")
parser.add_argument("output_path", help="Destination export", default="output")
args = parser.parse_args()

file_name = args.src
output_path = args.output_path
chunk_length_ms = int(args.chunk_length_ms)
threshold = float(args.threshold)
recognition = Recognition(100)

source_snare = AudioSegment.from_wav("./source/snare2.wav")
source_kick = AudioSegment.from_wav("./source/kick2.wav")

audio = AudioSegment.from_file(file_name, "wav")
audio = audio.set_frame_rate(16000)
audio = audio.set_sample_width(2)
audio = audio.split_to_mono()
audio = audio[0]
audio = normalize.normalize(audio, 0)
chunks = make_chunks(audio, chunk_length_ms)  # Make chunks of one sec

threshold_trigger = False
temporary_raw_sample = []
current_time = 0
current_temporary_raw_sample_time = 0

output_snare_positions = []
output_kick_positions = []
output_kicksnare_positions = []

for i, chunk in enumerate(chunks):
    current_time += chunk_length_ms
    raw_audio_data = chunk.raw_data
    sound = AudioSegment(data=b''.join([raw_audio_data]), sample_width=2, frame_rate=16000, channels=1)
    if sound.max_dBFS > threshold:
        threshold_trigger = True

    if threshold_trigger:
        if not temporary_raw_sample:
            current_temporary_raw_sample_time = current_time
        temporary_raw_sample.append(raw_audio_data)
        if len(temporary_raw_sample) >= 10:
            temporary_sound = AudioSegment(data=b''.join(temporary_raw_sample), sample_width=2, frame_rate=16000,
                                           channels=1)
            temp_filename = './temp.wav'
            temporary_sound.export(temp_filename, format='wav')
            prediction = recognition.predict(temp_filename)
            if prediction == 'snare':
                output_snare_positions.append(current_temporary_raw_sample_time)
            if prediction == 'kick':
                output_kick_positions.append(current_temporary_raw_sample_time)
            if prediction == 'kicksnare':
                output_kicksnare_positions.append(current_temporary_raw_sample_time)
            temporary_raw_sample = []
            threshold_trigger = False

output_master = AudioSegment(data=b''.join([]), sample_width=2, frame_rate=44100, channels=2)
output_master += AudioSegment.silent(len(audio))

if len(output_snare_positions) > 0:
    output_snare = AudioSegment(data=b''.join([]), sample_width=2, frame_rate=44100, channels=2)
    output_snare += AudioSegment.silent(len(audio))
    for position in output_snare_positions:
        output_snare = output_snare.overlay(source_snare, position=position)
        output_master = output_master.overlay(source_snare, position=position)
    output_snare.export(os.path.join('./', output_path, 'snare.wav'), format='wav')

if len(output_kick_positions) > 0:
    output_kick = AudioSegment(data=b''.join([]), sample_width=2, frame_rate=44100, channels=2)
    output_kick += AudioSegment.silent(len(audio))
    for position in output_kick_positions:
        output_kick = output_kick.overlay(source_kick, position=position)
        output_master = output_master.overlay(source_kick, position=position)
    output_kick.export(os.path.join('./', output_path, 'kick.wav'), format='wav')

if len(output_kicksnare_positions) > 0:
    output_kicksnare = AudioSegment(data=b''.join([]), sample_width=2, frame_rate=44100, channels=2)
    output_kicksnare += AudioSegment.silent(len(audio))
    for position in output_kicksnare_positions:
        output_kicksnare = output_kicksnare.overlay(source_snare, position=position)
        output_kicksnare = output_kicksnare.overlay(source_kick, position=position)
        output_master = output_master.overlay(source_snare, position=position)
        output_master = output_master.overlay(source_kick, position=position)
    output_kicksnare.export(os.path.join('./', output_path, 'kicksnare.wav'), format='wav')


output_master.export('./output/master.wav', format='wav')
print("Success")
