import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from src.config import CFG
from src.data_loader import create_dataloader
import os

# Set paths
BASE_PATH = "/app/birdclef-2024"
csv_file = os.path.join(BASE_PATH, "filtered_data_with_labels.csv")

# Load the feature extractor and model
extractor = AutoFeatureExtractor.from_pretrained(CFG.feature_extractor_name)
model = AutoModelForAudioClassification.from_pretrained(CFG.model_name, num_labels=9)  # 9 is the number of classes, adjust accordingly
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Create DataLoaders
train_loader = create_dataloader(csv_file=csv_file, base_path=BASE_PATH, batch_size=CFG.batch_size, extractor=extractor, test=False)
val_loader = create_dataloader(csv_file=csv_file, base_path=BASE_PATH, batch_size=CFG.batch_size, extractor=extractor, test=True)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(CFG.num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')  # Move to GPU if available
        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{CFG.num_epochs}, Training Loss: {total_loss / len(train_loader)}")

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{CFG.num_epochs}, Validation Loss: {val_loss / len(val_loader)}")
