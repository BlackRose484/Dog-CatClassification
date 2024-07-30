import torch
import torchvision
from torchvision import models

from torch import nn

class EfficientNetB0(nn.Module):
  def __init__(self, input_shape, output_shape, hidden_units):
    super().__init__()
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.model = models.efficientnet_b0(pretrained=True)
    self.model.classifier = nn.Sequential(
        nn.Linear(in_features= self.model.classifier[1].in_features,
                 out_features=128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=128,
                 out_features=output_shape),
    )

  def forward(self, x):
    return self.model(x)


def create_effnetb0_model(num_classes: int = 2,
                          seed: int = 42):

    model = EfficientNetB0(input_shape=2,
                            output_shape=2,
                            hidden_units=10)

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    transforms = weights.transforms()

    # Freeze all layers in base model
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)

    return model, transforms
