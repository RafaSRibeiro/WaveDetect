import os
import wave
import recognition
import audioinfo

file_name = './test/clipe3.wav'
wavefile = wave.open(file_name, 'r')
data = []
audioinfo.info(file_name)
quit()

frames_after_predict = 0
# for i in range(0, length):
buffer_position = 0
buffer = 320
time_position = 0
while buffer_position < wavefile.getnframes():
    frames = wavefile.readframes(buffer)
    data.append(frames)
    time_position += 10
    frames_after_predict += 1
    if frames_after_predict > 10:
        filename = './temp.wav'
        waveFile = wave.open(filename, 'wb')
        waveFile.setnchannels(1)
        waveFile.setsampwidth(2)
        waveFile.setframerate(16000)
        final_index = int(buffer_position / buffer)
        initial_index = final_index - 10
        waveFile.writeframes(b''.join(data[initial_index:final_index]))
        waveFile.close()
        prediction = recognition.predict(filename)
        if prediction == 'snare':
            print(prediction, time_position - 100)
            os.rename(filename, './output/snare'+str(time_position)+'.wav')
            frames_after_predict = 0
        if prediction == 'kick':
            print(prediction, time_position - 100)
            os.rename(filename, './output/kick' + str(time_position) + '.wav')
            frames_after_predict = 0
            # frames = []
    buffer_position += buffer
