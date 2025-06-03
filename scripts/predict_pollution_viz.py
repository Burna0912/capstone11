import cv2
import os
from glob import glob

# 경로 설정
image_dir = "data/image"  # 예: 원본 이미지 경로
label_dir = "data/label"  # 예: YOLO 형식 라벨 경로
output_dir = "outputs/visualized"

# 클래스 매핑
class_map = {
    0: "Clean",
    1: "Polluted"
}

# 색상 매핑
color_map = {
    0: (255, 0, 0),     # 파랑 (Clean)
    1: (0, 0, 255),     # 빨강 (Polluted)
}

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 이미지 파일 리스트
image_files = glob(os.path.join(image_dir, "*.jpg"))  # 확장자 필요시 변경

for image_path in image_files:
    filename = os.path.basename(image_path)
    label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))
    
    # 이미지 로드
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # 라벨 존재 확인
    if not os.path.exists(label_path):
        print(f"[경고] 라벨 파일 없음: {label_path}")
        continue

    # 라벨 읽기
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            
            # YOLO → 픽셀 좌표
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            # 색상과 라벨
            color = color_map.get(cls, (0, 255, 0))
            label = class_map.get(cls, f"Class {cls}")

            # 바운딩 박스
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 텍스트 박스 배경
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(image, (x1, y1 - text_h - 6), (x1 + text_w, y1), color, -1)

            # 텍스트 그리기
            cv2.putText(image, label, (x1, y1 - 4), font, font_scale, (255, 255, 255), font_thickness)

    # 저장
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, image)
    print(f"[완료] 시각화 저장됨: {save_path}")
