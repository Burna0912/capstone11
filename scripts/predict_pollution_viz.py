import os
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# 설정
MODEL_PATH = "outputs/efficientnet_pollution.pth"
IMAGE_DIR = "data/image"
LABEL_DIR = "data/label"
CROPPED_DIR = "data/cropped"
RESULT_DIR = "results"
CLASS_NAMES = ["clean", "polluted"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 디렉토리 생성
os.makedirs(RESULT_DIR, exist_ok=True)

# 모델 로딩
model = EfficientNet.from_name('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 이미지 루프
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, img_name.replace(".jpg", ".txt"))

    if not os.path.exists(label_path):
        continue

    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size

    with open(label_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        cls, cx, cy, w, h = map(float, line.strip().split())
        x1 = int((cx - w / 2) * width)
        y1 = int((cy - h / 2) * height)
        x2 = int((cx + w / 2) * width)
        y2 = int((cy + h / 2) * height)

        # 잘린 이미지 찾기
        crop_basename = f"{img_name.replace('.jpg','')}_{i}.jpg"
        crop_path = os.path.join(CROPPED_DIR, crop_basename)
        if not os.path.exists(crop_path):
            continue

        crop = Image.open(crop_path).convert("RGB")
        input_tensor = transform(crop).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            label = CLASS_NAMES[pred.item()]
            color = "red" if label == "polluted" else "green"

        # 박스: 더 두껍게
        draw.rectangle([x1, y1, x2, y2], outline=color, width=10)

        # 텍스트: 상태 표시 + 배경
        text = f"Pollution: {label}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_height - 2), text, fill="white", font=font)

    save_path = os.path.join(RESULT_DIR, f"pred_{img_name}")
    image.save(save_path)
    print(f"[{img_name}] → 시각화 저장 완료")
