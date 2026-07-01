import torch
import torch.nn as nn
import evaluate
import numpy as np
from transformers import ViTForImageClassification


class ViTClassifier(nn.Module):
    """
    Wrapper around a HF ViT model for image classification.

    Args:
        model_name:         HF identifier of the pre-trained model
        num_classes:        Number of target classes
        dropout_rate:       Dropout rate 
        freeze_backbone:    If True, all layers except the classification head are kept frozen
    """

    def __init__(self,
                 model_name: str = "google/vit-base-patch16-224",
                 num_classes: int = 6,
                 dropout_rate: float = 0.2,
                 freeze_backbone: bool = False):
        super().__init__()

        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
            )
        
        # Adapt classifier head
        hidden_dim = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes)
            )

        if freeze_backbone:
            # Freeze all layers except the classification head
            for name, param in self.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function to prepare a batch of data for the ViT model.
        """
        pixel_values = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }
    
    @staticmethod
    def compute_metrics(pred):
        """
        Compute metrics for the ViT model.
        """
        metric = evaluate.load("accuracy")
        logits, labels = pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    
if __name__ == "__main__":
    # Define a ViT classifier for testing
    vit = ViTClassifier(
        model_name="google/vit-base-patch16-224",
        num_classes=6,
        dropout_rate=0.2,
        freeze_backbone=True
        )
    print(vit.model.classifier)

    # Check input and output shapes
    BATCH_SIZE   = 8
    IMAGE_SHAPE  = (3, 224, 224)
    dummy = torch.randn(BATCH_SIZE, *IMAGE_SHAPE)
    outputs = vit.model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Logits shape: {outputs.logits.shape}")