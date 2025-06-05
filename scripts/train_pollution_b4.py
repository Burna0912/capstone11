import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

# 하이퍼파라미터
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["clean", "polluted"]

# 🧾 데이터셋 정의
class PollutionDataset(Dataset):
    def __init__(self, file_list_path):
        with open(file_list_path, 'r') as f:
            lines = f.read().splitlines()

        self.samples = []
        for line in lines:
            img_path, label_path, pollution_status = line.split(',')
            label = 0 if pollution_status == 'clean' else 1
            self.samples.append((img_path, label))

        self.transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label

# 📂 경로
data_root = "data/original"
train_file = os.path.join(data_root, "train_list.txt")
val_file = os.path.join(data_root, "val_list.txt")
test_file = os.path.join(data_root, "test_list.txt")

# 📤 데이터로더
train_loader = DataLoader(PollutionDataset(train_file), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(PollutionDataset(val_file), batch_size=BATCH_SIZE)
test_loader = DataLoader(PollutionDataset(test_file), batch_size=BATCH_SIZE)

# 🧠 모델 정의
model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.DEFAULT')
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(DEVICE)

# ⚙️ 손실함수, 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 🔁 학습 루프
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = correct / len(train_loader.dataset)
    print(f"Train Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

    # 📈 검증
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
    val_acc = correct / len(val_loader.dataset)
    print(f"Validation Accuracy: {val_acc:.4f}")

# 🧪 테스트
model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
test_acc = correct / len(test_loader.dataset)
print(f"Test Accuracy: {test_acc:.4f}")

# 💾 저장
torch.save(model.state_dict(), "pollution_b4_model.pth")
