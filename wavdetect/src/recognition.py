import os
import pathlib
from plot import get_wavform
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

model = tf.keras.models.load_model('../../jupyter/work/model')

sample_files = ['./test/noise.wav', './test/kick.wav', './test/snare.wav']

sample_size = 1500
data_dir = pathlib.Path('../../jupyter/work/dataset/mono')
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
AUTOTUNE = tf.data.AUTOTUNE


def get_spectrogram(waveform):
    # Padding for files with less than 3000 samples
    if [sample_size] - tf.shape(waveform) > 0:
        zero_padding = tf.zeros([sample_size] - tf.shape(waveform), dtype=tf.float32)
    else:
        zero_padding = tf.zeros(0, dtype=tf.float32)

    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform[0:sample_size], zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)

    return spectrogram


def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)
    ax.axis('off')


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
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds


for sample_file in sample_files:
    waveform = get_wavform(sample_file)
    spectrogram = get_spectrogram(waveform)

    sample_ds = preprocess_dataset([str(sample_file)])
    for spectrogram, label in sample_ds.batch(1):
      prediction = model.predict(spectrogram)
      print(commands[prediction.argmax(axis=1)[0]])
      # plt.bar(commands, tf.nn.softmax(prediction[0]))
      # plt.show()
