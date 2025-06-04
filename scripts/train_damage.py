import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

# 🔧 경로 추가 (CAPSTONE11/utils 경로 인식)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.transforms import get_transforms

# ✅ 설정
DATA_DIR = 'data/damage'
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 데이터 로드
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, 'train'),
    transform=get_transforms('train')
)
val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, 'val'),
    transform=get_transforms('val')
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ✅ Vision Transformer 모델
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)
model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ✅ 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ✅ 학습 루프
if __name__ == "__main__":
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        # ✅ 검증
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(train_dataset)
        val_acc = correct / total * 100
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f} | Val Accuracy = {val_acc:.2f}%")

    # ✅ 모델 저장
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/vit_damage.pth")
    print("✅ ViT 훼손도 모델 저장 완료: outputs/vit_damage.pth")
