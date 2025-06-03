import os
import torch
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet

# 모델 경로와 클래스 정의
MODEL_PATH = "outputs/efficientnet_pollution.pth"
CLASS_NAMES = ['clean', 'polluted']  # 클래스 순서 주의

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_name('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# 예측할 이미지 폴더
image_dir = "test_images"  # 추론용 이미지가 담긴 폴더 (직접 준비)
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
               if fname.lower().endswith(('.jpg', '.png', '.jpeg'))]

# 추론 시작
for img_path in image_paths:
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        label = CLASS_NAMES[pred.item()]
        print(f"{os.path.basename(img_path)} → 예측 결과: {label}")
