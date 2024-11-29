import torch.nn as nn
from transformers import AutoModel

class BirdClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super(BirdClassifier, self).__init__()
        self.ast = AutoModel.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.ast.config.hidden_size, num_classes)

    def forward(self, x):
        x = self.ast(x).last_hidden_state.mean(dim=1)  # Pooling
        x = self.classifier(x)
        return x