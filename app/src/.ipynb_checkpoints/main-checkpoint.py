import torch
from torch.optim import Adam
from data_loader import build_dataloaders
from model import BirdClassifier
from trainer import train_one_epoch, validate

def main(data_dir, num_classes, batch_size, epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataloaders
    train_loader, valid_loader = build_dataloaders(data_dir, batch_size)

    # Initialize model and optimizer
    model = BirdClassifier(num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        train_loss, train_f1 = train_one_epoch(model, optimizer, train_loader, device)
        valid_loss, valid_f1 = validate(model, valid_loader, device)

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} - "
              f"Valid Loss: {valid_loss:.4f}, Valid F1: {valid_f1:.4f}")

if __name__ == "__main__":
    main(data_dir="/birdclef-2024", num_classes=10, batch_size=32, epochs=10, lr=1e-4)