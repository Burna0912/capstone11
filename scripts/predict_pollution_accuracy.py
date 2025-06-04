import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# 설정
IMAGE_DIR = "data/pollute/image"
LABEL_DIR = "data/pollute/label"
MODEL_PATH = "outputs/efficientnet_pollution.pth"
CLASS_NAMES = ["clean", "polluted"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 불러오기
model = EfficientNet.from_name('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 정확도 계산 변수
total = 0
correct = 0

# 이미지 루프
for filename in os.listdir(IMAGE_DIR):
    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    label_path = os.path.join(LABEL_DIR, filename.replace(".jpg", ".txt"))

    if not os.path.exists(label_path):
        continue

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        true_label = int(parts[0])
        _, cx, cy, bw, bh = map(float, parts)

        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        crop = image.crop((x1, y1, x2, y2))
        crop_tensor = transform(crop).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(crop_tensor)
            pred = torch.argmax(output, dim=1).item()

        total += 1
        if pred == true_label:
            correct += 1

# 정확도 출력
accuracy = correct / total * 100 if total > 0 else 0
print(f"✅ 총 바운딩박스 수: {total}")
print(f"✅ 맞춘 개수: {correct}")
print(f"🎯 정확도 (Accuracy): {accuracy:.2f}%")
