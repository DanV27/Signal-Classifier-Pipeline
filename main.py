from datasets import Dataset, load_dataset, Audio, load_from_disk
import numpy as np
import matplotlib.pyplot as plt
import io
import librosa
import whisper


def run():
    ds = load_dataset("MLCommons/peoples_speech", "clean", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    small = ds.take(200)

    small_ds = Dataset.from_list(list(small))
    small_ds.save_to_disk("data/peoples_speech_sample")

def load():
    ds = load_from_disk("data/peoples_speech_sample")
    sample = ds[50]


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


def transcribe(audio):
    audio = load()


if __name__ == "__main__":
    audio_sample = load()

