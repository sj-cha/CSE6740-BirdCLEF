import torch
from transformers import AutoModelForAudioClassification

def load_model(model_name, num_labels, device='cuda'):
    # Sanity check: Print the number of labels
    print(f"Loading model with num_labels = {num_labels}")
    
    # Load the model with the specified number of labels
    model = AutoModelForAudioClassification.from_pretrained(model_name, torch_dtype=torch.float16)
    
    # Check and adjust the classifier head if needed
    if hasattr(model, 'classifier'):
        # For most models, the classifier will be a layer or head to adjust
        classifier = model.classifier
        if isinstance(classifier, torch.nn.Linear):
            if classifier.out_features != num_labels:
                print(f"Adjusting classifier output from {classifier.out_features} to {num_labels}")
                model.classifier = torch.nn.Linear(in_features=classifier.in_features, out_features=num_labels)
        else:
            print("The classifier is not a simple linear layer. You might need a custom adjustment.")
    else:
        print("Model does not have a 'classifier' attribute. You may need to manually modify the head.")
    
    model.to(device)
    return model
