import wave
import recognition

wavefile = wave.open('./test/snare4.wav', 'r')

data = []

frames_after_predict = 0
# for i in range(0, length):
buffer_position = 0
buffer = 160
while buffer_position < wavefile.getnframes():
    frames = wavefile.readframes(buffer)
    data.append(frames)
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
            print(prediction)
            frames_after_predict = 0
            # frames = []
    buffer_position += buffer
