import tempfile
import json
import os
import re

from datasets import Dataset, load_dataset, Audio, load_from_disk
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
import whisper

SR = 16000  # whisper.load_audio always returns 16 kHz mono
CACHE_PATH = "data/transcripts.json"

_ds = None
_model = None
_cache = None


def run():
    ds = load_dataset("MLCommons/peoples_speech", "clean", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    small = ds.take(2000)

    small_ds = Dataset.from_list(list(small))
    small_ds.save_to_disk("data/peoples_speech_sample")


def load(row):
    '''
    uses global _ds so the dataset only loads from disk once
    '''
    global _ds
    if _ds is None:
        _ds = load_from_disk("data/peoples_speech_sample")
    return _ds[row]


def get_model():
    '''
    loads Whisper once and reuses it.
    old version did `model = ...` (a local), so _model stayed None
    and the model reloaded on every single file.
    '''
    global _model
    if _model is None:
        _model = whisper.load_model("base")
    return _model


# ---------- transcript cache ----------

def _key(audio):
    # stable id per clip, so the cache survives reordering/reloading the dataset
    return audio.get("id") or audio["audio"]["path"]


def load_cache():
    global _cache
    if _cache is None:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH) as f:
                _cache = json.load(f)
        else:
            _cache = {}
    return _cache


def save_cache():
    if _cache is None:
        return
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_cache, f)
    os.replace(tmp, CACHE_PATH)  # swap in one step so a crash can't corrupt the cache


def is_cached(audio):
    return _key(audio) in load_cache()


# ---------- audio ----------

def decode(audio):
    with tempfile.NamedTemporaryFile(suffix=".mp3") as temp:
        temp.write(audio['audio']['bytes'])
        temp.flush()
        return whisper.load_audio(temp.name)  # float32 array, 16 kHz


def graph_plot(y, sr=SR):
    t = np.arange(len(y)) / sr  # sample index -> seconds

    plt.figure(figsize=(10, 4))
    plt.plot(t, y)
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


def mel_plot(y, sr=SR, n_mels=64, size=2.24, dpi=100):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=128, n_mels=n_mels)
    mel_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(size, size), dpi=dpi)
    librosa.display.specshow(mel_db, sr=sr, hop_length=128, cmap='viridis', ax=ax)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def transcribe_time(audio, y=None):
    '''
    uses Whisper to get every word in the audio and its timestamps.
    results are cached by clip id in data/transcripts.json, so each
    file only ever gets transcribed ONCE, no matter how many words
    you search for later.

    :param audio: dataset row
    :param y: already-decoded audio (optional, skips a second decode)
    :return: {word: ["0.52s ->0.81s", ...]}
    '''
    cache = load_cache()
    key = _key(audio)
    if key in cache:
        return cache[key]

    if y is None:
        y = decode(audio)

    # pass the array straight in instead of making Whisper decode the mp3 again
    result = get_model().transcribe(y, word_timestamps=True, fp16=False)  # fp16=False since no gpu

    text_dict = {}
    for segment in result["segments"]:
        for w in segment["words"]:
            word = re.sub(r"[^\w\s]", "", w["word"].strip().lower())
            word_timestamp = f"{w['start']:.2f}s ->{w['end']:.2f}s"
            text_dict.setdefault(word, []).append(word_timestamp)

    cache[key] = text_dict
    return text_dict


def get_word(search_word, audio, y=None):
    '''
    returns every timestamp where search_word is said in the audio
    (empty list if it's not in there).
    '''
    search_word = re.sub(r"[^\w\s]", "", search_word.strip().lower())

    text_dict = transcribe_time(audio, y)
    hits = text_dict.get(search_word, [])

    if hits:
        print(f"The word '{search_word}' is in the audio at these times: {hits}")
    else:
        print(f"The word '{search_word}' is not in this audio!")

    return hits


def get_slice(audio, word_timestamps, pad=0.1, y=None):
    '''
    takes audio and word_timestamps from get_word and returns
    the clip of when that word was said.
    pass y if you already decoded the audio.
    '''
    if y is None:
        y = decode(audio)
    start, end = [float(x) for x in re.findall(r'\d+\.\d+', word_timestamps[0])]

    s = max(0, int((start - pad) * SR))
    e = min(len(y), int((end + pad) * SR))
    return y[s:e]


def save_mel(search_word, range_count, out_dir="data/spectrograms/about"):
    '''
    runs through range_count files and saves a mel spectrogram for
    every time search_word is said.
    first run transcribes + caches every file (the slow part, one time only).
    after that it's just lookups, and only files with the word get decoded.
    '''
    os.makedirs(out_dir, exist_ok=True)

    try:
        for i in range(range_count):
            if i and i % 25 == 0:
                save_cache()  # checkpoint so a crash doesn't lose progress

            audio_sample = load(i)
            print(f"AUDIO: {i} ----")

            # only decode up front if we have to transcribe anyway
            y = None if is_cached(audio_sample) else decode(audio_sample)

            word_timestamps = get_word(search_word, audio_sample, y)
            if not word_timestamps:
                continue

            if y is None:
                y = decode(audio_sample)

            for j, ts in enumerate(word_timestamps):
                clip = get_slice(audio_sample, [ts], y=y)
                fig = mel_plot(clip)
                # _{j} so two hits in the same file don't overwrite each other
                fig.savefig(f"{out_dir}/{search_word}_{i}_{j}.png", dpi=100, pad_inches=0)
                plt.close(fig)
    finally:
        save_cache()


if __name__ == "__main__":

    #run()

    #for w in ['about']:
        #save_mel(w, 2000)



    APP_DIR = "data/spectrograms/about"

    # Counts ONLY files, ignoring subfolders
    file_count = sum(1 for entry in os.scandir(APP_DIR) if entry.is_file())

    print(f"Total files: {file_count}")








