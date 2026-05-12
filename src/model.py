import torch
import torch.nn as nn
import torchvision.models as models

# ─── FFT Feature Extractor ───────────────────────────────
class FFTFeatureExtractor(nn.Module):
    def __init__(self, img_size=224, output_size=256):
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Sequential(
            nn.Linear(img_size * img_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        gray      = x.mean(dim=1)
        fft       = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.log(torch.abs(fft_shift) + 1)
        magnitude = magnitude.view(magnitude.size(0), -1)
        return self.fc(magnitude)


# ─── Spatial Feature Extractor ───────────────────────────
class SpatialFeatureExtractor(nn.Module):
    def __init__(self, output_size=256):
        super().__init__()
        eff           = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.features = eff.features
        self.pool     = eff.avgpool
        self.fc       = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ─── Verifai Detector ────────────────────────────────────
class VerifaiDetector(nn.Module):
    def __init__(self, img_size=224):
        super().__init__()
        self.spatial_branch = SpatialFeatureExtractor(256)
        self.fft_branch     = FFTFeatureExtractor(img_size, 256)
        self.classifier     = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        s = self.spatial_branch(x)
        f = self.fft_branch(x)
        return self.classifier(torch.cat([s, f], dim=1))


# keep old name as alias for backward compatibility
MisinformationDetector = VerifaiDetector