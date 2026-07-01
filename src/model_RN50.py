import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 image classifier.

    Args:
        num_classes:        Number of target classes
        dropout_rate:       Dropout rate
        freeze_backbone:    If True, keep all ResNet layers frozen to fine-tune only the head)
    """

    def __init__(self, num_classes: int,
                 dropout_rate: float,
                 freeze_backbone: bool = False):
        super().__init__()

        # Load the pretrained vanilla ResNet‑50 weights
        self.backbone = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")

        # Fully‑connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=2048,
                      out_features=num_classes,
                      bias=True)
        )

        if freeze_backbone:
            # Freeze all layers except the classification head
            for name, param in self.backbone.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Input shape : (B, 3, 256, 256)
        Output shape: (B, num_classes)
        """
        logits = self.backbone(x)
        return logits

if __name__ == "__main__":
    # Define a vanilla ResNet-50 classifier for testing
    model = ResNet50Classifier(
        num_classes=10,
        dropout_rate=0.2,
        freeze_backbone=False
        )
    print(model.backbone.fc)

    # Check input and output shapes
    BATCH_SIZE   = 8
    IMAGE_SHAPE  = (3, 256, 256)
    dummy = torch.randn(BATCH_SIZE, *IMAGE_SHAPE)
    logits = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Logits shape: {logits.shape}")