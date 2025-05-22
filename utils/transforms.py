# utils/transforms.py
from torchvision import transforms

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize([0.5], [0.5])
    ])
