import torch
import sys
from PIL import Image
from torchvision import transforms
import os

sys.path.insert(0, 'src')
from model import VerifaiDetector

model = VerifaiDetector(img_size=224)
model.load_state_dict(torch.load('models/verifai_best.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# test on 5 known REAL images
real_path = 'data/raw/test/REAL'
fake_path = 'data/raw/test/FAKE'

print("=== REAL IMAGES ===")
for img_name in os.listdir(real_path)[:5]:
    img    = Image.open(os.path.join(real_path, img_name)).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
    pred = "REAL" if probs[0] > probs[1] else "FAKE"
    print(f"{img_name}: {pred} (REAL={probs[0]*100:.1f}% FAKE={probs[1]*100:.1f}%)")

print("\n=== FAKE IMAGES ===")
for img_name in os.listdir(fake_path)[:5]:
    img    = Image.open(os.path.join(fake_path, img_name)).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
    pred = "REAL" if probs[0] > probs[1] else "FAKE"
    print(f"{img_name}: {pred} (REAL={probs[0]*100:.1f}% FAKE={probs[1]*100:.1f}%)")