import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor
from sklearn.model_selection import train_test_split
from src.config import CFG
import numpy as np

def prepare_metadata(csv_file, base_path, test_size=0.2):
    """
    Prepares the metadata by cleaning file paths, balancing the dataset, and splitting into train and test sets.
    
    Args:
        csv_file (str): Path to the CSV file containing metadata.
        base_path (str): The base directory where audio files are stored.
        test_size (float): The proportion of the dataset to include in the test split (default 0.2).
    
    Returns:
        pd.DataFrame, pd.DataFrame: Processed training and test DataFrames with filepaths and labels.
    """
    # Load the metadata CSV file
    df = pd.read_csv(csv_file).drop(columns="y")  # Drop unwanted columns

    # Ensure correct file path format
    df["filename"] = df["filename"].str.split("/").apply(lambda x: "/".join(x[-2:]))  # Fix relative paths
    df["filepath"] = df["filename"].apply(lambda x: os.path.join(base_path, "train_audio", x))  # Full file path

    # Balance the dataset: sample equal examples for each class
    df = df.groupby('primary_label').sample(n=df['primary_label'].value_counts().min(), random_state=CFG.seed)

    # Convert labels to numeric categories
    df["target"] = df["primary_label"].astype("category").cat.codes  # Convert labels to numeric

    # Split into train and test datasets
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['primary_label'], random_state=CFG.seed)
    
    return train_df, test_df

class BirdCLEFDataset(Dataset):
    def __init__(self, df, extractor):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata and file paths.
            extractor (transformers.AutoFeatureExtractor): Feature extractor from Hugging Face.
        """
        self.df = df
        self.extractor = extractor
        self.target_len = CFG.audio_len 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["filepath"]

        # Load audio file
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform[0].numpy()  # Convert to 1D NumPy array

        # Ensure correct sample rate
        if sr != CFG.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=CFG.sample_rate)(torch.tensor(waveform)).numpy()

        if len(waveform) > CFG.audio_len:
            start_idx = torch.randint(0, len(waveform) - CFG.audio_len, (1,)).item()
            waveform = waveform[start_idx:start_idx + CFG.audio_len]

        # Extract features using the processor
        inputs = self.extractor(waveform, sampling_rate=CFG.sample_rate, return_tensors="pt", padding=True)
        # Return the audio features and label
        label = torch.tensor(row["target"], dtype=torch.long)
        return inputs.input_values.squeeze(0), label

def create_dataloader(csv_file, base_path, batch_size, extractor, shuffle=True, test=False):
    """
    Create a DataLoader for the BirdCLEF dataset (train or test).
    
    Args:
        csv_file (str): Path to the CSV file containing metadata.
        base_path (str): The base directory where audio files are stored.
        batch_size (int): Batch size for DataLoader.
        extractor (transformers.AutoFeatureExtractor): Feature extractor for AST model.
        shuffle (bool): Whether to shuffle the dataset.
        test (bool): If True, create a test DataLoader instead of train DataLoader.
    
    Returns:
        DataLoader: PyTorch DataLoader.
    """
    train_df, test_df = prepare_metadata(csv_file, base_path)  # Prepare the metadata

    # Select train or test DataFrame
    df = test_df if test else train_df
    dataset = BirdCLEFDataset(df, extractor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=CFG.num_workers)