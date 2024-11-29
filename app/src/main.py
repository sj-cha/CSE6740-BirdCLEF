import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoFeatureExtractor
from src.config import CFG
from src.data_loader import create_dataloader
from src.model import load_model
from train import train  # Importing train function from train.py


def main():
    # Set paths
    BASE_PATH = "/app/birdclef-2024"
    master_path = os.path.join(BASE_PATH, "master.csv")
    csv_file = os.path.join(BASE_PATH, "filtered_data_with_labels.csv")

    # Load the feature extractor for AST model
    print("Loading feature extractor...")
    extractor = AutoFeatureExtractor.from_pretrained(CFG.feature_extractor_name)

    # Load and prepare filtered data
    print("Loading filtered data...")
    df = pd.read_csv(csv_file).drop(columns='y')
    df["filename"] = df["filename"].str.split('/').apply(lambda x: '/'.join(x[-2:]))
    df.index = df["filename"]

    # Split into train and test datasets
    print("Splitting data into train and test sets...")
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['primary_label'], random_state=42)

    # Load and prepare master data
    print("Loading master data...")
    master = pd.read_csv(master_path).iloc[:, 1:]
    master.index = master["filename"]

    # Align train and test sets with master data using their index
    df_train = master.loc[df_train.index].reset_index(drop=True)
    df_test = master.loc[df_test.index].reset_index(drop=True)

    classes = df_train['primary_label'].unique()
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=df_train['primary_label'])
    
    # Compute num_labels dynamically based on the unique values in 'primary_label'
    num_labels = len(df_train['primary_label'].unique())
    print(f"Number of unique labels: {num_labels}")

    # Load the model with the correct num_labels
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(CFG.model_name, num_labels=num_labels, device=device)

    # Move model to GPU and convert to float16
    model = model.to(device)  # Convert model to float16
    print(f"Model is running on: {device}")
    
    # Check train and test sizes and unique labels
    print(f"Train set size: {len(df_train)} samples")
    print(f"Test set size: {len(df_test)} samples")
    print(f"Unique labels in train set: {len(df_train['primary_label'].unique())}")
    print(f"Unique labels in test set: {len(df_test['primary_label'].unique())}")

    # Create DataLoader for training
    print("Creating DataLoader for training data...")
    train_loader = create_dataloader(csv_file=csv_file, base_path=BASE_PATH, batch_size=CFG.batch_size, extractor=extractor)

    # Create DataLoader for testing
    print("Creating DataLoader for testing data...")
    test_loader = create_dataloader(csv_file=csv_file, base_path=BASE_PATH, batch_size=CFG.batch_size, extractor=extractor, shuffle=False, test=True)


    # Convert class weights to a tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device) # Ensure it's on the correct device
    
    # Start training by calling train function from train.py
    train(model, train_loader, test_loader, device, CFG, class_weights_tensor)


if __name__ == "__main__":
    main()
