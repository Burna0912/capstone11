from pathlib import Path

# 경로 설정
images_dir = Path("data/images")
labels_dir = Path("data/labels")

# images 폴더 내 output_*.jpg → output 제거한 이름 추출
image_names = {f.stem for f in images_dir.glob("output_*.jpg")}
image_names_cleaned = {name.replace("output_", "", 1) for name in image_names}

renamed, deleted = [], []

for txt_file in labels_dir.glob("*.txt"):
    stem = txt_file.stem

    # 이미지로부터 유도된 이름과 매칭되는지 확인
    if stem in image_names_cleaned:
        # 이름 변경: output_ 접두어 추가
        new_name = f"output_{stem}.txt"
        new_path = labels_dir / new_name
        txt_file.rename(new_path)
        renamed.append((txt_file.name, new_name))
    else:
        # 해당 이미지가 없으면 삭제
        txt_file.unlink()
        deleted.append(txt_file.name)

print(f"[✓] 리네이밍 완료: {len(renamed)}개")
print(f"[🗑] 삭제된 label 파일: {len(deleted)}개")
