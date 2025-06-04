import os
import shutil
import random
from pathlib import Path

# 정답 라벨된 이미지 폴더
source_dir = Path("data/categorized_damage")
classes = ["intact", "damage"]

# 타겟 경로
train_dir = Path("data/damage/train")
val_dir = Path("data/damage/val")
val_ratio = 0.5


for cls in classes:
    cls_source = source_dir / cls
    images = list(cls_source.glob("*.jpg"))
    random.shuffle(images)

    val_size = int(len(images) * val_ratio)
    val_images = images[:val_size]
    train_images = images[val_size:]

    # 폴더 생성
    (train_dir / cls).mkdir(parents=True, exist_ok=True)
    (val_dir / cls).mkdir(parents=True, exist_ok=True)

    # 복사
    for img in train_images:
        shutil.copy(img, train_dir / cls / img.name)

    for img in val_images:
        shutil.copy(img, val_dir / cls / img.name)

print("✅ train/val 폴더 분리 완료")
