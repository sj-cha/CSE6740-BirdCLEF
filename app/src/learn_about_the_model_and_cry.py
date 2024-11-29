import torch
from transformers import AutoModelForAudioClassification

# Load the model
model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-large-xlsr-53", torch_dtype=torch.float16)

# Print high-level layers (e.g., encoder, projector, classifier)
for name, module in model.named_modules():
    if 'encoder' in name or 'projector' in name or 'classifier' in name:
        print(name, module)
