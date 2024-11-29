import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoFeatureExtractor
from src.config import CFG
from src.data_loader import create_dataloader
from src.model import load_model

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

    # Compute num_labels dynamically based on the unique values in 'primary_label'
    num_labels = len(df_train['primary_label'].unique())
    print(f"Number of unique labels: {num_labels}")

    # Load the model with the correct num_labels
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(CFG.model_name, num_labels=num_labels, device=device)

    # Move model to GPU and convert to float16
    model = model.to(device).half()  # Convert model to float16
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

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    print("Starting training...")
    for epoch in range(CFG.num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).half(), labels.to(device)  # Cast inputs to float16

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).logits
            loss = criterion(outputs, labels.float())  # Convert loss labels to float32

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{CFG.num_epochs}, Loss: {total_loss / len(train_loader)}")

    # Evaluate on the test set
    print("Evaluating model on the test set...")
    model.eval()  # Set model to evaluation mode
    total_correct = 0
    total_samples = 0
    total_f1_macro = 0
    total_f1_weighted = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device).half(), labels.to(device)  # Cast inputs to float16
            outputs = model(inputs).logits
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            # Calculate F1 scores
            f1_macro = f1_score(labels.cpu(), predicted.cpu(), average='macro')
            f1_weighted = f1_score(labels.cpu(), predicted.cpu(), average='weighted')
            total_f1_macro += f1_macro
            total_f1_weighted += f1_weighted

    accuracy = total_correct / total_samples
    f1_macro_avg = total_f1_macro / len(test_loader)
    f1_weighted_avg = total_f1_weighted / len(test_loader)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {f1_macro_avg:.4f}")
    print(f"Test Weighted F1: {f1_weighted_avg:.4f}")

    # Optionally, save the model
    print("Saving the model...")
    torch.save(model.state_dict(), 'birdclef_model.pth')

if __name__ == "__main__":
    main()
