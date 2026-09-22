import tempfile
from datasets import Dataset, load_dataset, Audio, load_from_disk
import numpy as np
import matplotlib.pyplot as plt
import io
import librosa, librosa.display
import whisper
import pprint as pp
import re
import os


SR = 16000  # whisper.load_audio always returns 16 kHz mono
_ds = None
_model = None


def run():
    ds = load_dataset("MLCommons/peoples_speech", "clean", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    small = ds.take(200)

    small_ds = Dataset.from_list(list(small))
    small_ds.save_to_disk("data/peoples_speech_sample")

def load(row):
    '''
    now uses global variable _ds so it doesnt have to load the data everytime for every word when iterated through
    :param row:
    :return:
    '''
    global _ds
    if _ds is None:
        _ds = load_from_disk("data/peoples_speech_sample")
    return _ds[row]


    return sample
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


def transcribe_time(audio):
    '''
    this function uses Whisper to take in raw audio bytes
     and outputs the transcription to the audio,
      including each word and their timestamp!
       and rn it compares the original text to the one
        whisper made.

    :param audio:
    :return:
    '''


    global _model
    if _model is None:
        model = whisper.load_model("base")
    #model = whisper.load_model("base")
    raw_bytes = audio['audio']['bytes']
    with tempfile.NamedTemporaryFile(suffix=".mp3") as temp:
        temp.write(raw_bytes)
        temp.flush()
        result = model.transcribe(temp.name, word_timestamps=True, fp16=False) #fp16 is because i dont got gpu, so it suppresses a warning

    #pp.pprint(result)
    text_dict = {}
    for segment in result["segments"]:
        for words in segment["words"]:
            word = words["word"]
            word = word.strip()
            word = word.lower()
            word = re.sub(r"[^\w\s]","", word)

            # okay make value hold a list for mutiple time stamps if there are multiple instances of the word!
            word_timestamp = f"{words['start']:.2f}s ->{words['end']:.2f}s"
            time_list = [word_timestamp] #incase there are multiple word instances in audio

            if word in text_dict:
                text_dict[word].append(word_timestamp)
            else:
                text_dict[word] = time_list

    #pp.pprint(text_dict)
    return text_dict


def get_word(search_word, audio):
    '''
    takes a search_word and audio file, transcribes in the
     function and returns whether or not that word is in the audio.
      if there's multiple instances of that word in the audio,
       it'll give them as well.
    :param search_word:
    :param audio:
    :return:
    '''

    search_word = search_word.lower()
    search_word = search_word.strip()
    search_word = re.sub(r"[^\w\s]", "", search_word)

    text_dict = transcribe_time(audio)


    if search_word in text_dict:
        if len(text_dict[search_word]) > 0:
            print(f"The word '{search_word}' is in the audio at these times: {text_dict[search_word]}")
    else:
        print(f"The word '{search_word}' is not in this audio!")

    return text_dict.get(search_word,[])

def get_slice(audio, word_timestamps, pad=0.1):
    '''
    takes audio and word_timestamps from get_word,
     and returns a mel_plot of when that word was said!

    :param audio:
    :param word_timestamps:
    :param pad:
    :return: clip
    '''
    y = decode(audio)
    start, end = [float(x) for x in re.findall(r'\d+\.\d+', word_timestamps[0])]

    s = max(0, int((start - pad) * SR))
    e = min(len(y), int((end + pad) * SR))
    clip = y[s:e]


    return clip

def save_mel():
    """
    Just a dummy function, inside is what i used to save a
    specific word mel spectrogram. runs through 200 and saves them
    to a specific folder.
    :return:
    """

    os.makedirs("data/spectrograms", exist_ok=True)

    for i in range(200):
        audio_sample = load(i)
        print(f"AUDIO: {i} ----")
        word_timestamps = get_word('the', audio_sample)

        for j, ts in enumerate(word_timestamps):
            clip = get_slice(audio_sample, [ts])
            fig = mel_plot(clip)
            fig.savefig(f"data/spectrograms/the_spec/the_{i}_{j}.png", dpi=100, pad_inches=0)
            plt.close(fig)





if __name__ == "__main__":
    x = 10



