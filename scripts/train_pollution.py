import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader

# 경로 설정
train_dir = "data/train"
val_dir = "data/val"
pretrained_weights_path = "weights/efficientnet-b0-355c32eb.pth"
MODEL_NAME = 'efficientnet-b0'
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001

# 데이터 전처리
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
}

# 데이터 로딩
train_dataset = datasets.ImageFolder(train_dir, transform=transform['train'])
val_dataset = datasets.ImageFolder(val_dir, transform=transform['val'])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 로드 및 가중치 적용
model = EfficientNet.from_name(MODEL_NAME)
state_dict = torch.load(pretrained_weights_path)
model.load_state_dict(state_dict)
model._fc = nn.Linear(model._fc.in_features, NUM_CLASSES)  # 분류기 교체

# 학습 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 학습 루프
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"[Epoch {epoch+1}] Training Loss: {running_loss / len(train_loader):.4f}")

# 모델 저장
output_dir = os.path.join(os.path.dirname(__file__), "../outputs")
os.makedirs(output_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(output_dir, "efficientnet_pollution.pth"))

print("학습 완료 및 모델 저장됨.")

