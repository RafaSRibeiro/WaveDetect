import os
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

import tensorflow as tf

model = tf.keras.models.load_model('../../jupyter/work/model3')

sample_files = ['./test/noise.wav', './test/kick.wav', './test/snare.wav', './file.wav', './test/snare2.wav',
                './test/temp.wav']

data_dir = pathlib.Path('../../jupyter/work/dataset/mono')
commands = np.array(tf.io.gfile.listdir(str(data_dir)))


def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 1600
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [1600] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=32)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=-1)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    return output_ds.map(get_spectrogram_and_label_id, num_parallel_calls=tf.data.AUTOTUNE)


def predict(file):
    # audio_file = file
    # split_audio = AudioSegment.from_wav(audio_file)
    # split_audio = split_audio[0:100]
    # split_audio_file = 'temp.wav'
    # split_audio.export(split_audio_file, format="wav")
    # stereo_audio = AudioSegment.from_file(split_audio_file, format="wav")
    # stereo_audio = stereo_audio.set_frame_rate(16000)
    # mono_audios = stereo_audio.split_to_mono()
    # mono_audios[0].export('temp.wav', format="wav")

    sample_ds = preprocess_dataset([str(file)])
    for spectrogram, label in sample_ds.batch(1):
        prediction = model.predict(spectrogram)
        return commands[prediction.argmax(axis=1)[0]]
        # plt.bar(commands, tf.nn.softmax(prediction[0]))
        # plt.title(f'Predictions for "{commands[label[0]]}"')
        # plt.show()

# predict(sample_files[3])
