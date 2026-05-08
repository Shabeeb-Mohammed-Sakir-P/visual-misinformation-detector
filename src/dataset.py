import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ─── Constants ───────────────────────────────────────────
IMG_SIZE = 32       # CIFAKE images are 32x32
BATCH_SIZE = 32
NUM_WORKERS = 0     # Keep 0 for Windows

# ─── Transforms ──────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ─── Dataset Class ───────────────────────────────────────
class CIFAKEDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        root_dir: path to data/raw
        split: 'train' or 'test'
        """
        self.transform = transform
        self.samples = []
        self.labels = []

        for label, cls in enumerate(['REAL', 'FAKE']):
            cls_dir = os.path.join(root_dir, split, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append(os.path.join(cls_dir, img_name))
                    self.labels.append(label)  # REAL=0, FAKE=1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ─── DataLoader Factory ──────────────────────────────────
def get_dataloaders(data_dir):
    train_dataset = CIFAKEDataset(data_dir, split='train',
                                   transform=train_transforms)
    test_dataset  = CIFAKEDataset(data_dir, split='test',
                                   transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    return train_loader, test_loader