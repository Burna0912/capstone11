# scripts/crop_from_boxed_image.py

import cv2
import os

def extract_objects_from_box_image(img_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    index = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 50 or h < 50 or w/h > 5 or h/w > 5:
            continue

        cropped = image[y:y+h, x:x+w]
        out_name = os.path.join(
            out_dir,
            f"{os.path.splitext(os.path.basename(img_path))[0]}_crop_{index}.jpg"
        )
        cv2.imwrite(out_name, cropped)
        index += 1

    print(f"[✓] {index}개 객체 추출 완료 → {os.path.basename(img_path)}")

def extract_all_from_folder(input_folder="input_images", out_dir="data/unlabeled"):
    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png"))]
    for f in files:
        full_path = os.path.join(input_folder, f)
        extract_objects_from_box_image(full_path, out_dir)

if __name__ == "__main__":
    extract_all_from_folder()
