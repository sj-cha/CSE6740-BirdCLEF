import torch
from transformers import AutoModelForAudioClassification
from torch import nn

def load_model(model_name, num_labels, device='cuda'):
    # Load the pre-trained model
    print(f"Loading model with num_labels = {num_labels}")
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    
    # Check the current classifier
    if hasattr(model, 'classifier'):
        classifier = model.classifier
        print("Original classifier structure:", classifier)
        
        # If the classifier is a simple linear layer, extract its in_features and replace it
        if isinstance(classifier, nn.Module):  # It should be an nn.Module
            in_features = classifier.dense.in_features
            print(f"Classifier input features: {in_features}")
            
            # Replace classifier with a new one using FP32
            model.classifier = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_labels)
            )
        else:
            print("The classifier is not a simple linear layer. Replacing with a custom classifier.")
            # If classifier is not a simple layer, replace it entirely
            model.classifier = nn.Sequential(
                nn.LayerNorm(classifier.in_features),
                nn.Linear(classifier.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_labels)
            )
    else:
        print("Model does not have a 'classifier' attribute. You may need to manually modify the head.")
        # You can create a custom classifier if 'classifier' is not present

    # Move model to the appropriate device
    model.to(device)
    
    return model
