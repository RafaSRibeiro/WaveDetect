from pydub import AudioSegment


def normalize(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.max_dBFS
    return sound.apply_gain(change_in_dBFS)


def normalize_by_file(file, target_dBFS):
    sound = AudioSegment.from_file(file, "wav")
    return normalize(sound, target_dBFS)
