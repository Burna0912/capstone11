import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

# 클래스 이름 (ImageFolder 기준)
class_names = ["intact", "damage"]

# 모델 로딩
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)
model.heads.head = nn.Linear(model.heads.head.in_features, len(class_names))
model.load_state_dict(torch.load("outputs/vit_damage.pth", map_location=DEVICE))
model.eval().to(DEVICE)

# transform (val 기준)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 예측 함수
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_prob, pred_class = torch.max(probs, 1)

    class_label = class_names[pred_class.item()]
    confidence = top_prob.item() * 100
    return image_path, class_label, confidence

# 테스트 대상 폴더 또는 파일
TEST_DIR = "test_images"  # 예시 경로
test_images = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# 예측 실행
for img_path in test_images:
    path, label, score = predict_image(img_path)
    print(f"{os.path.basename(path)} → {label} ({score:.2f}%)")
