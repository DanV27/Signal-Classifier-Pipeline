# Signal Classifier Pipeline

An end-to-end audio pipeline that turns raw speech into spectrogram images and trains a model to recognize spoken words (keyword spotting).

## How it works

1. **Data**: Streams 200 samples from the [MLCommons People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech) "clean" dataset and saves them locally.
2. **Word timestamps**: Uses OpenAI Whisper to transcribe each clip and find exactly when each word is spoken.
3. **Slicing**: Cuts out the audio for a target word (with a small padding).
4. **Spectrograms**: Converts each word clip into a mel spectrogram image with librosa.
5. **Training**: Fine-tunes a pretrained ResNet-50 on the spectrogram images (80/20 train/test split) to classify which word was said.

Current model classifies between the words **"the"** and **"about"**, hitting around **98% test accuracy**.

> Note: the dataset is imbalanced (276 "the" vs 26 "about" samples), so always guessing "the" would already score about 91%. Balancing the classes and adding more words is next.

## Project structure

```
main.py    # data download, Whisper transcription, word slicing, spectrogram generation
model.py   # ResNet-50 training and evaluation on the spectrograms
data/      # dataset sample + generated spectrograms (one folder per word)
```

## Tech stack

Python, PyTorch, torchvision (ResNet-50), OpenAI Whisper, librosa, Hugging Face Datasets, NumPy, Matplotlib

## Getting started

```bash
git clone https://github.com/DanV27/Signal-Classifier-Pipeline.git
cd Signal-Classifier-Pipeline
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision openai-whisper librosa datasets numpy matplotlib pillow
```

Whisper also needs [ffmpeg](https://ffmpeg.org/) installed.

Then:
1. Run `run()` in `main.py` to download the dataset sample.
2. Run `save_mel()` in `main.py` to generate spectrograms for a word.
3. Run `python model.py` to train and evaluate the classifier.

## Coming soon

I'm currently building a **React front end** (backed by an API) to showcase what the pipeline can do, like uploading audio, seeing the spectrograms, and getting live word predictions.

## Author

Daniel Valenzuela · [github.com/DanV27](https://github.com/DanV27)
