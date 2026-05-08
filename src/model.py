import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

# ─── FFT Feature Extractor ───────────────────────────────
class FFTFeatureExtractor(nn.Module):
    def __init__(self, output_size=128):
        super(FFTFeatureExtractor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(32 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: (batch, 3, 32, 32)
        # convert to grayscale by averaging channels
        gray = x.mean(dim=1)  # (batch, 32, 32)

        # apply FFT
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.log(torch.abs(fft_shift) + 1)

        # flatten and pass through FC layers
        magnitude = magnitude.view(magnitude.size(0), -1)  # (batch, 1024)
        out = self.fc(magnitude)
        return out


# ─── EfficientNet Spatial Branch ─────────────────────────
class SpatialFeatureExtractor(nn.Module):
    def __init__(self, output_size=128):
        super(SpatialFeatureExtractor, self).__init__()
        # load pretrained EfficientNet-B0
        efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')

        # remove the final classifier
        self.features = efficientnet.features
        self.pool = efficientnet.avgpool

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ─── Fusion Classifier ───────────────────────────────────
class MisinformationDetector(nn.Module):
    def __init__(self):
        super(MisinformationDetector, self).__init__()
        self.spatial_branch = SpatialFeatureExtractor(output_size=128)
        self.fft_branch = FFTFeatureExtractor(output_size=128)

        # fusion: concatenate both branches (128 + 128 = 256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)  # 2 classes: REAL, FAKE
        )

    def forward(self, x):
        spatial_features = self.spatial_branch(x)
        fft_features = self.fft_branch(x)

        # fuse both features
        combined = torch.cat([spatial_features, fft_features], dim=1)
        output = self.classifier(combined)
        return output