import os
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

# 설정
IMAGE_DIR = "data/image"
LABEL_DIR = "data/label"
SAVE_DIR = "results_damage"
MODEL_PATH = "outputs/vit_damage.pth"
CLASS_NAMES = ["intact", "damage"]

# 디바이스
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로딩
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)
model.heads.head = nn.Linear(model.heads.head.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 시각화 예측
def predict_and_draw(image_path, label_path, save_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    w, h = image.size

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        crop = image.crop((x1, y1, x2, y2))
        crop_tensor = transform(crop).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(crop_tensor)
            prob = torch.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)

        label = CLASS_NAMES[pred.item()]
        score = conf.item() * 100
        text = f"{label} ({score:.1f}%)"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), text, fill="red", font=font)

    image.save(save_path)
    print(f"{os.path.basename(image_path)} → 저장 완료 → {save_path}")

# 실행
os.makedirs(SAVE_DIR, exist_ok=True)
for filename in os.listdir(IMAGE_DIR):
    if filename.endswith(".jpg"):
        img_path = os.path.join(IMAGE_DIR, filename)
        label_path = os.path.join(LABEL_DIR, filename.replace(".jpg", ".txt"))
        if os.path.exists(label_path):
            save_path = os.path.join(SAVE_DIR, filename)
            predict_and_draw(img_path, label_path, save_path)
