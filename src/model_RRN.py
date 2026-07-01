import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import timm

class RobustRailNet(nn.Module):
    """
    RobustRailNet: ResNet-50 + SwinTransformerV2-B hybrid with CBAM attention.

    Args:
        num_classes:   Number of output classes 
        hidden_dim:    Number of units in the first fully connected layer of the classification head
        dropout_rate:  Dropout probability in the classification head
        pretrained:    If True, load ImageNet-pretrained weights for the ResNet-50 and Swin Transformer branches
    """

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 512,
        dropout_rate: float = 0.5,
        pretrained: bool = True,
    ):
        super().__init__()

        self.gaussian = GaussianFilter(kernel_size=9, sigma=0.1, channels=3)

        self.resnet_branch = ResNetCBAMBranch(pretrained=pretrained)
        self.swin_branch   = SwinV2Branch(pretrained=pretrained)

        combined_features = self.resnet_branch.out_features + self.swin_branch.out_features
        self.head = ClassificationHead(
            in_features=combined_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gaussian(x)

        feat_resnet = self.resnet_branch(x)
        feat_swin   = self.swin_branch(x)

        combined = torch.cat([feat_resnet, feat_swin], dim=1)
        return self.head(combined)

class GaussianFilter(nn.Module):
    """
    Fixed Gaussian blur applied channel-wise.

    Args:
        kernel_size:  Size of the Gaussian kernel (must be odd)
        sigma:        Standard deviation of the Gaussian
        channels:     Number of input channels
    """
    def __init__(self, kernel_size: int = 9, sigma: float = 0.1, channels: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels

        kernel = self._gaussian_kernel(kernel_size, sigma)
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        self.register_buffer("weight", kernel)

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_2d = torch.outer(g, g)
        kernel_2d /= kernel_2d.sum()
        return kernel_2d.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        return F.conv2d(x, self.weight, padding=pad, groups=self.channels)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.

    Args:
        in_channels:  Number of input channels
        reduction:    Reduction ratio for the MLP
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg = x.mean(dim=[2, 3])
        mx  = x.amax(dim=[2, 3])
        scale = torch.sigmoid(self.shared_mlp(avg) + self.shared_mlp(mx))
        return x * scale.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Args:
        kernel_size:  Size of the convolutional kernel (must be odd)
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        concat = torch.cat([avg, mx], dim=1)
        scale = torch.sigmoid(self.conv(concat))
        return x * scale


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) introduced by Woo et al. (2018).
    
    Args:
        in_channels:    Number of input channels
        reduction:      Reduction ratio for the channel attention
        spatial_kernel: Size of the convolutional kernel for spatial attention (must be odd)
    """

    def __init__(self, in_channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class ResNetCBAMBranch(nn.Module):
    """
    ResNet-50 Branch with CBAM injected after each residual layer group.
    
    Args:
        pretrained:  If True, load ImageNet-pretrained weights
    """

    # Output channel widths for ResNet-50 layers
    _LAYER_CHANNELS = [256, 512, 1024, 2048]

    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Stem + pool (shared entry)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)

        # Each residual group paired with its CBAM
        self.layer1 = base.layer1
        self.cbam1  = CBAM(self._LAYER_CHANNELS[0])

        self.layer2 = base.layer2
        self.cbam2  = CBAM(self._LAYER_CHANNELS[1])

        self.layer3 = base.layer3
        self.cbam3  = CBAM(self._LAYER_CHANNELS[2])

        self.layer4 = base.layer4
        self.cbam4  = CBAM(self._LAYER_CHANNELS[3])

        # Average pooling to produce a 2048-d feature vector
        self.avgpool = base.avgpool
        self.out_features = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))

        x = self.avgpool(x)
        return torch.flatten(x, 1)
    

class SwinV2Branch(nn.Module):
    """
    Swin Transformer V2-Base.

    Args:
        pretrained:  If True, load ImageNet-pretrained weights
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            "swinv2_base_window8_256",
            pretrained=pretrained,
            num_classes=0,
        )
        self.out_features = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ClassificationHead(nn.Module):
    """
    Classification head for RobustRailNet.
        - Batch Normalization (1D)
        - Fully Connected Layer
        - ReLU
        - Dropout
        - Fully Connected Layer
    
    Args:
        in_features:   Number of input features
        num_classes:   Number of output classes
        hidden_dim:    Number of units in the first fully connected layer
        dropout_rate:  Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


if __name__ == "__main__":
    # Define a RobustRailNet architecture for testing
    model = RobustRailNet(
        num_classes=6,
        hidden_dim=512,
        dropout_rate=0.4,
        pretrained=False
        )
    print(model.head)

    # Check input and output shapes
    BATCH_SIZE   = 8
    IMAGE_SHAPE  = (3, 256, 256)
    dummy = torch.randn(BATCH_SIZE, *IMAGE_SHAPE)
    logits = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Logits shape: {logits.shape}")