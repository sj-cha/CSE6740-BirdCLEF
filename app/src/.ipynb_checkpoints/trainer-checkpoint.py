import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(outputs.argmax(dim=1).cpu().numpy())

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return running_loss / len(train_loader), macro_f1

def validate(model, valid_loader, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            running_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return running_loss / len(valid_loader), macro_f1