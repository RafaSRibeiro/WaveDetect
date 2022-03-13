import os
import recognition
from pydub import AudioSegment
from pydub.utils import make_chunks

file_name = './test/clipe3.wav'

audio = AudioSegment.from_file(file_name, "wav")
chunk_length_ms = 10  # pydub calculates in millisecond
chunks = make_chunks(audio, chunk_length_ms)  # Make chunks of one sec

threshold = False
temporary_raw_sample = []
current_time = 0
current_temporary_raw_sample_time = 0
# Convert chunks to raw audio data which you can then feed to HTTP stream
for i, chunk in enumerate(chunks):
    current_time += chunk_length_ms
    raw_audio_data = chunk.raw_data
    sound = AudioSegment(data=b''.join([raw_audio_data]), sample_width=2, frame_rate=16000, channels=1)
    if sound.max_dBFS > -6:
        threshold = True

    if threshold:
        if not temporary_raw_sample:
            current_temporary_raw_sample_time = str(current_time / 1000)
        temporary_raw_sample.append(raw_audio_data)
        if len(temporary_raw_sample) >= 30:
            temporary_sound = AudioSegment(data=b''.join(temporary_raw_sample), sample_width=2, frame_rate=16000, channels=1)
            temp_filename = './temp.wav'
            temporary_sound.export(temp_filename, format='wav')
            prediction = recognition.predict(temp_filename)
            if prediction == 'snare':
                print(prediction, current_temporary_raw_sample_time)
                os.rename(temp_filename, './output/snare'+str(current_temporary_raw_sample_time)+'.wav')
                frames_after_predict = 0
            if prediction == 'kick':
                print(prediction, current_temporary_raw_sample_time)
                os.rename(temp_filename, './output/kick' + str(current_temporary_raw_sample_time) + '.wav')
                frames_after_predict = 0
            temporary_raw_sample = []
            threshold = False
