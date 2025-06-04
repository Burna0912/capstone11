import cv2
import os
from pathlib import Path

# 경로 설정
images_dir = Path("data/image")
labels_dir = Path("data/label")
output_dir = Path("data/cropped")
output_dir.mkdir(parents=True, exist_ok=True)

# 이미지 파일 순회
for image_path in images_dir.glob("*.jpg"):
    label_path = labels_dir / f"{image_path.stem}.txt"
    
    # 라벨 없으면 스킵
    if not label_path.exists():
        continue

    # 이미지 불러오기
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"이미지 로딩 실패: {image_path}")
        continue
    h, w = image.shape[:2]

    # 라벨 읽고 crop
    with open(label_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_id, x_center, y_center, bw, bh = map(float, parts)

        # YOLO 정규화 좌표 → 실제 픽셀 좌표
        x1 = int((x_center - bw / 2) * w)
        y1 = int((y_center - bh / 2) * h)
        x2 = int((x_center + bw / 2) * w)
        y2 = int((y_center + bh / 2) * h)

        # 경계 조건 보정
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        output_path = output_dir / f"{image_path.stem}_{i}.jpg"
        cv2.imwrite(str(output_path), crop)
