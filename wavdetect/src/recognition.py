import os
import numpy as np
import tensorflow as tf


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


class Recognition(object):

    def __init__(self, audio_length_ms=100):
        self.audio_length_ms = audio_length_ms
        self.model = tf.keras.models.load_model('../../jupyter/work/models/model' + str(self.audio_length_ms) + 'ms')
        self.commands = np.array(['kick', 'noise', 'snare', 'kicksnare'])
        self.sample_len = int(audio_length_ms / 50) * 800

    def get_spectrogram(self, waveform):
        # Zero-padding para uma forma de onda de áudio com menos de 16.000 amostras.
        input_len = self.sample_len
        waveform = waveform[:input_len]
        zero_padding = tf.zeros([input_len] - tf.shape(waveform), dtype=tf.float32)
        # Transmita o dtype dos tensores de forma de onda para float32.
        waveform = tf.cast(waveform, dtype=tf.float32)
        # Concatene a forma de onda com `zero_padding`, o que garante que todos os
        # clipes de áudio tenham a mesma duração.
        equal_length = tf.concat([waveform, zero_padding], 0)
        # Converta a forma de onda em um espectrograma por meio de um STFT.
        spectrogram = tf.signal.stft(equal_length, frame_length=80, frame_step=16)
        # Obtenha a magnitude do STFT.
        spectrogram = tf.abs(spectrogram)
        # Adicione uma dimensão `channels`, para que o espectrograma possa ser usado como
        # dados de entrada semelhantes a imagens com camadas de convolução
        # (que esperam forma (`batch_size`, `height`, `width`, `channels`).
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    def get_spectrogram_and_label_id(self, audio, label):
        spectrogram = self.get_spectrogram(audio)
        label_id = tf.argmax(label == self.commands)
        return spectrogram, label_id

    def preprocess_dataset(self, files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        return output_ds.map(self.get_spectrogram_and_label_id, num_parallel_calls=tf.data.AUTOTUNE)

    def predict(self, file):
        sample_ds = self.preprocess_dataset([str(file)])
        for spectrogram, label in sample_ds.batch(1):
            prediction = self.model.predict(spectrogram)
            return self.commands[prediction.argmax(axis=1)[0]]
