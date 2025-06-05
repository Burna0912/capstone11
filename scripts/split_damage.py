import json
import random
from pathlib import Path

json_list_path = Path("data/json.txt")
json_paths = json_list_path.read_text(encoding="utf-8").splitlines()

INTACT_KEYWORD = "원형"
CLEAN_KEYWORD = "오염없음"

damage_data, pollute_data = [], []

def make_paths(file_name):
    img_path = f"/data/original/images/{file_name}"
    lbl_path = f"/data/original/labels/{file_name.replace('.jpg', '.txt')}"
    return img_path, lbl_path

for json_path in json_paths:
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        file_name = data["IMAGE_INFO"]["FILE_NAME"]
        ann_list = data["ANNOTATION_INFO"]
        if not ann_list:
            print(f"⚠️  Empty annotation: {json_path}")
            continue

        ann = ann_list[0]
        damage_type = ann.get("DAMAGE", "")
        dirtiness = ann.get("DIRTINESS", "")

        img_path, lbl_path = make_paths(file_name)

        damage_class = "intact" if damage_type == INTACT_KEYWORD else "damage"
        pollute_class = "clean" if dirtiness == CLEAN_KEYWORD else "polluted"

        damage_data.append((img_path, lbl_path, damage_class))
        pollute_data.append((img_path, lbl_path, pollute_class))

    except Exception as e:
        print(f"❌ Error parsing {json_path}: {e}")

def split_data(data):
    random.shuffle(data)
    n = len(data)
    return {
        "train": data[:int(n*0.6)],
        "val": data[int(n*0.6):int(n*0.8)],
        "test": data[int(n*0.8):]
    }

def save_split(name, split):
    base = Path(f"data/{name}")
    base.mkdir(parents=True, exist_ok=True)
    for split_name, items in split.items():
        with open(base / f"{split_name}.txt", "w", encoding="utf-8") as f:
            for img, lbl, cls in items:
                f.write(f"{img},{lbl},{cls}\n")

save_split("damage", split_data(damage_data))
save_split("pollute", split_data(pollute_data))
