import torch
from torch.optim.lr_scheduler import StepLR  # Import the scheduler
from sklearn.metrics import f1_score
from tqdm import tqdm  # For progress bar
from torch.cuda.amp import autocast, GradScaler
import os

def train(model, train_loader, test_loader, device, CFG, class_weights_tensor):
    model = model.float()  # Ensure the model is in FP32 (no mixed precision)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    criterion = torch.nn.CrossEntropyLoss(weight = class_weights_tensor)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)


    best_val_loss = float('inf')
    best_model_wts = None  # To store the best model weights

    for epoch in range(CFG.num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.num_epochs}", unit="batch")):
            inputs, labels = inputs.to(device).float(), labels.to(device).long()  # Ensure inputs are FP32

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)

            loss.backward()

            # Check for NaNs in gradients
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"NaN or inf detected in gradients of {name} at batch {batch_idx}!")

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{CFG.num_epochs} - Training Loss: {avg_train_loss:.4f}")

        # After each epoch, evaluate on the validation set and print validation metrics
        val_loss, val_accuracy, val_f1_macro, val_f1_weighted = evaluate(model, test_loader, device, CFG)
        
        print(f"Epoch {epoch + 1}/{CFG.num_epochs}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Macro F1: {val_f1_macro:.4f}")
        print(f"Validation Weighted F1: {val_f1_weighted:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()  # Save the model weights
            
            print(f"New best model found at epoch {epoch+1}. Saving model...")

            # Save the model to a file
            model_save_path = os.path.join(CFG.model_save_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(best_model_wts, model_save_path)
            print(f"Model saved to {model_save_path}")
        scheduler.step()


def evaluate(model, test_loader, device, CFG):
    model = model.float()  # Ensure the model is in FP32
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_f1_macro = 0
    total_f1_weighted = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        with tqdm(test_loader, desc="Evaluating", unit="batch") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device).float(), labels.to(device).long()  # Ensure inputs are FP32
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

                f1_macro = f1_score(labels.cpu(), predicted.cpu(), average='macro')
                f1_weighted = f1_score(labels.cpu(), predicted.cpu(), average='weighted')
                total_f1_macro += f1_macro
                total_f1_weighted += f1_weighted

                pbar.set_postfix(accuracy=total_correct / total_samples, 
                                 f1_macro=total_f1_macro / (pbar.n + 1),
                                 f1_weighted=total_f1_weighted / (pbar.n + 1))

    accuracy = total_correct / total_samples
    f1_macro_avg = total_f1_macro / len(test_loader)
    f1_weighted_avg = total_f1_weighted / len(test_loader)
    val_loss = total_loss / len(test_loader)

    return val_loss, accuracy, f1_macro_avg, f1_weighted_avg
