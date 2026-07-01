import torch
from torch.utils.data import Dataset, random_split
from pathlib import Path
from PIL import Image
import pandas as pd
from torchvision import transforms
import xml.etree.ElementTree as ET
from omegaconf import DictConfig
from typing import Dict, List
import matplotlib.pyplot as plt
import json

from model_RN50 import ResNet50Classifier
from model_RRN import RobustRailNet
from model_ViT import ViTClassifier

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ImageClassificationDataset(Dataset):
    """
    Dataset for image classification tasks.

    Args:
        pairs:              List of tuples [(image path, label)]
        transform:          torchvision transforms to apply to each image
    """
    def __init__(self, pairs, transform=None):
        self.samples = pairs
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # load image
        image = Image.open(img_path).convert('RGB')

        # apply transform
        if self.transform:
            image = self.transform(image)

        return image, label
    
class EarlyStopping:
    """
    Early stopping for model fine-tuning.

    Args:
        patience:       Number of epochs to wait for improvement before stopping.
        delta:          Minimum change in the monitored quantity to qualify as an improvement.
        verbose:        If True, prints messages when the model improves.
        ckpt_path:      Path to save the best model checkpoint.
    """
    def __init__(self, patience=5, delta=0.0, verbose=False, ckpt_path="best_model.pt"):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.ckpt_path = ckpt_path
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
            torch.save(model, self.ckpt_path)
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.6f}. Model checkpoint saved to {self.ckpt_path}.")
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")
    
def load_data(
        config: DictConfig,
        ) -> tuple[
            torch.utils.data.Dataset,
            torch.utils.data.Dataset,
            torch.utils.data.Dataset
            ]:
    """
    Returns train, validation and test datasets according to configuration.
    """
    data_path_1 = Path(config.data.data_path_1)
    data_path_2 = Path(config.data.data_path_2)
    json_path = Path(config.data.json_path)

    # Load label to index mapping
    label2idx, _ = load_label_mappings(json_path)

    image_transforms = transforms.Compose([
        transforms.Resize((config.model.target_height, config.model.target_width), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), 
    ])

    # Load pairs
    match config.data.name:
        case 'original':
            pairs = load_v1_data(data_path_1)
        case 'baseline':
            train_pairs = load_v2_data(data_path_1 / 'train')
            test_pairs = load_v2_data(data_path_1 / 'test')
            pairs = train_pairs+test_pairs
        case 'synthetic' :
            pairs = load_diff_data(data_path_1)
        case 'synthetic+':
            synthetic_data = load_diff_data(data_path_1)
            original_data = load_v2_data( data_path_2 / 'train') + load_v2_data(data_path_2 / 'test')
            pairs = synthetic_data+original_data

    # Map label to index
    pairs = [(p, label2idx[l]) for p, l in pairs if l in label2idx]
    print(f"  Loaded {len(pairs)} images from the {config.data.name} dataset.")

    # Create dataset
    full_dataset = ImageClassificationDataset(pairs, transform=image_transforms)

    # Define split sizes dataset for train, val and test dataset
    torch.manual_seed(config.training.seed)
    dataset_size = len(full_dataset)
    train_size = int(0.7*dataset_size)
    val_size = int(0.15*dataset_size)
    test_size = dataset_size-train_size-val_size

    # Perform data splitting
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.training.seed)
    )

    return train_dataset, val_dataset, test_dataset

def load_model(config: DictConfig) -> torch.nn.Module:
    """
    Returns model architecture according to configuration.
    """
    match config.model.name:
        case 'rrn':
            model = RobustRailNet(
                num_classes = config.data.num_classes,
                hidden_dim = config.model.hidden_dim,
                dropout_rate = config.model.dropout_rate,
                pretrained = config.model.pretrained,
            )
            print("RobustRailNet architecture:")
        case 'rn50':
            model = ResNet50Classifier(
                num_classes = config.data.num_classes,
                dropout_rate = config.model.dropout_rate,
                freeze_backbone = config.model.freeze_backbone
            )
            print("ResNet-50 architecture:")
        case 'vit':
            model = ViTClassifier(
                model_name_or_path = "google/vit-base-patch16-224",
                num_classes = config.data.num_classes,
                freeze_backbone = config.model.freeze_backbone
            )
            print("ViT classifier architecture:")
    return model

def load_history(hist_path: str) -> Dict[str, List[float]]:
    """
    Load loss history from .pt file.
    """    
    history = torch.load(Path(hist_path), weights_only=False)

    # Example: plot train & val loss together
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="Train loss", marker="o")
    plt.plot(epochs, history["val_loss"],   label="Val loss",   marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.show()

    # Plot test loss as a horizontal line (since it’s a single value)
    test_loss = history["test_loss"][0]
    plt.axhline(test_loss, color="red", linestyle="--", label=f"Test loss = {test_loss:.4f}")
    plt.legend()
    plt.show()

    return history
    
def get_object_name(xml_path: Path) -> str:
    """
    Reads xml-file and returns object name.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    obj = root.find('object')
    name = obj.find('name')
    return name.text.strip()

def load_diff_data(base_dir: Path):
    """
    Loads diffusion-based dataset and returns a list of tupels [(image_path, label_str)].
    """
    pairs = []

    for class_dir in sorted(base_dir.iterdir()):
        label = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            pairs.append((img_path, label))

    return pairs

def load_v2_data(base_dir: Path):
    """
    Searches base_dir for *.jpg and corresponding *.xml and returns a list of tupels [(image_path, label_str)].
    """
    pairs = []
    img_paths = sorted(base_dir.rglob('*.jpg'))

    for img_path in img_paths:
        xml_path = img_path.with_suffix('.xml')
        if not xml_path.is_file():
            raise FileNotFoundError(f'No annotation found for {img_path}')
        label = get_object_name(xml_path)
        pairs.append((img_path, label))

    return pairs

def load_v1_data(base_dir: Path):
    """
    Searches base_dir for *.jpg and corresponding label in classes.xlsx
    Returns a list of tupels [(image_path, label_str), ...].
    """
    pairs = []
    classes_df = pd.read_excel(base_dir / Path('classes.xlsx'), engine="openpyxl", header=None, names=["Class"])

    for row in classes_df.itertuples():
        img_idx = row.Index + 1
        img_path = base_dir / Path(f"1 Images/{img_idx}.jpg")
        label = row.Class
        if img_idx != 132:
            pairs.append((img_path, label))

    return pairs

def load_label_mappings(json_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    """
    Load the JSON file that contains a mapping
    and return both directions:
        - label2idx : {label (str) : idx (int)}
        - idx2label : {idx   (int) : label (str)}
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    label2idx = {str(k): int(v) for k, v in data["class_to_idx"].items()}
    idx2label = {idx: label for label, idx in label2idx.items()}

    return label2idx, idx2label

if __name__ == "__main__":
    print("This is a utility module.")