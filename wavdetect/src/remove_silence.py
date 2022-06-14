from pydub import AudioSegment


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    trim_ms = 0

    assert chunk_size > 0
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


def remove_initial_silence(sound, silence_threshold=-30):
    start_trim = detect_leading_silence(sound, silence_threshold)
    return sound[start_trim:]


def remove_initial_silence_by_file(sound):
    sound = AudioSegment.from_file("/path/to/file.wav", format="wav")
    return remove_initial_silence(sound)

