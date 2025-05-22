import torch
from torchvision.io import read_image
from efficientnet_pytorch import EfficientNet
from utils.transforms import get_transforms

CLASSES = ['intact', 'damaged']

def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = torch.nn.Linear(model._fc.in_features, 2)
    model.load_state_dict(torch.load("models/damage_efficientnet.pt", map_location=device))
    model.eval().to(device)

    image = read_image(image_path).float() / 255
    image = get_transforms()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()
        print(f"[damage] 예측 결과: {CLASSES[pred]}")

if __name__ == "__main__":
    predict("test.jpg")
