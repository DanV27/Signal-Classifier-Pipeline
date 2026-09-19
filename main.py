import tempfile

from datasets import Dataset, load_dataset, Audio, load_from_disk
import numpy as np
import matplotlib.pyplot as plt
import io
import librosa
import whisper
import pprint as pp
import re




def run():
    ds = load_dataset("MLCommons/peoples_speech", "clean", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    small = ds.take(200)

    small_ds = Dataset.from_list(list(small))
    small_ds.save_to_disk("data/peoples_speech_sample")

def load(row):
    ds = load_from_disk("data/peoples_speech_sample")
    sample = ds[row]


    return sample

def graph_plot(audio):
    raw_bytes = audio['audio']['bytes']

    # 1. Truncate trailing bytes if the buffer isn't a multiple of 4 bytes
    remainder = len(raw_bytes) % 4
    if remainder != 0:
        raw_bytes = raw_bytes[:-remainder]

    # 2. Convert bytes to int32 numpy array
    audio_data = np.frombuffer(raw_bytes, dtype=np.int32).copy()

    # 3. Optional: Normalize to a -1.0 to 1.0 float range
    # 32-bit signed ints have a max magnitude of 2**31
    audio_data_normalized = audio_data.astype(np.float32) / (2 ** 31)

    # Plot the usable waveform
    plt.figure(figsize=(10, 4))
    plt.plot(audio_data_normalized, linewidth=0.1, )
    plt.title("Audio Waveform Sample")
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


def mel_plot(audio):
    raw_bytes = audio['audio']['bytes']


    # Wrap your byte string in BytesIO
    file_like_object = io.BytesIO(raw_bytes)

    # Load into an audio waveform array
    y, sr = librosa.load(file_like_object, sr=None)

    # Compute and plot your Mel spectrogram normally
    S = librosa.feature.melspectrogram(y=y, sr=sr)

    mel_spect_db = librosa.power_to_db(S, ref=np.max)

    # 4. Plot the result
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel', fmax=sr / 2, cmap='viridis')

    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()


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
    model = whisper.load_model("base")
    raw_bytes = audio['audio']['bytes']
    with tempfile.NamedTemporaryFile(suffix=".mp3") as temp:
        temp.write(raw_bytes)
        temp.flush()
        result = model.transcribe(temp.name, word_timestamps=True)

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

    pp.pprint(text_dict)
    return text_dict


def get_word(search_word, audio):
    '''
    takes a search_word and audio file, transcribes in the
     function and returns whether or not that word is in the audio.
      if theres multiple instances of that word in the audio,
       it'll give them as well.
    :param search_word:
    :param audio:
    :return:
    '''

    search_word = search_word.lower()
    search_word = search_word.strip()
    search_word = re.sub(r"[^\w\s]", "", search_word)

    text_dict = transcribe_time(audio_sample)


    if search_word in text_dict:
        if len(text_dict[search_word]) > 0:
            print(f"The word '{search_word}' is in the audio at these times: {text_dict[search_word]}")
    else:
        print(f"{search_word} is not in this audio!")

if __name__ == "__main__":
    for i in range(0,5):
        audio_sample = load(i)
        print("-----------------------------------------------")
        print(f"Audio number: {i}")
        get_word("I'm", audio_sample)

