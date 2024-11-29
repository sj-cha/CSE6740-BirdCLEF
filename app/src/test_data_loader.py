import os
import librosa
import librosa.display as lid
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import numpy as np
from src.config import CFG

# Function to load audio
def load_audio(filepath):
    """
    Load an audio file with librosa.
    Args:
        filepath (str): Path to the audio file.
    Returns:
        audio (np.ndarray): Loaded audio data.
        sr (int): Sample rate.
    """
    audio, sr = librosa.load(filepath, sr=CFG.sample_rate)
    return audio, sr

# Function to get spectrogram
def get_spectrogram(audio):
    """
    Convert audio to a Mel Spectrogram.
    Args:
        audio (np.ndarray): Audio data.
    Returns:
        spec (np.ndarray): Mel spectrogram of the audio.
    """
    fmin = 20  # Minimum frequency (20Hz)
    fmax = 20000  # Maximum frequency (20kHz)

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=CFG.sample_rate,
        n_mels=256,
        n_fft=2048,
        hop_length=512,
        fmax=fmax,
        fmin=fmin,
    )
    spec = librosa.power_to_db(spec, ref=1.0)
    min_, max_ = spec.min(), spec.max()
    if max_ != min_:
        spec = (spec - min_) / (max_ - min_)  # Normalize to [0, 1]
    return spec

# Function to display audio, waveform, and spectrogram
def display_audio(row):
    """
    Display the audio waveform and spectrogram, and play the audio.
    Args:
        row (pd.Series): DataFrame row containing audio file information.
    """
    caption = f"Id: {row.filename}  | Rating: {row.rating}"
    audio, sr = load_audio(row.filepath)
    audio = audio[:CFG.audio_len]  # Clip to 10 seconds
    spec = get_spectrogram(audio)

    print("# Audio:")
    display(Audio(audio, rate=sr))  # This will display an audio player in Jupyter

    print("# Visualization:")
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, tight_layout=True)
    fig.suptitle(caption)

    # Waveform plot
    lid.waveshow(audio, sr=sr, ax=ax[0], color="blue")
    ax[0].set_title("Waveform")

    # Spectrogram plot
    # Use fmin and fmax from CFG or hard-code them if needed
    fmin = getattr(CFG, 'fmin', 20)  # Default to 20Hz if not present in CFG
    fmax = getattr(CFG, 'fmax', 20000)  # Default to 20kHz if not present in CFG

    lid.specshow(
        spec,
        sr=sr,
        hop_length=512,
        n_fft=2048,
        fmin=fmin,
        fmax=fmax,
        x_axis="time",
        y_axis="mel",
        cmap="coolwarm",
        ax=ax[1],
    )
    ax[1].set_title("Spectrogram")
    plt.show()