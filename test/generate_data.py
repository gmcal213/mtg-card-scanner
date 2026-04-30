import os
from PIL import Image
from pillow_heif import register_heif_opener
import subprocess
import time

register_heif_opener()

input_dir = "woodoffcenter"
output_dir = "images"

os.makedirs(output_dir, exist_ok=True)

TARGET = 1080  # long edge

card_map = {
    1: "aggressive_mammoth",
    2: "arresters_zeal",
    3: "audacious_thief",
    4: "cavern_of_souls",
    5: "chandra_novice_pyromancer",
    6: "cloudkin_seer",
    7: "corpse_knight",
    8: "duress",
    9: "empyrean_eagle",
    10: "iron_clad_krovod",
    11: "ironroot_warlord",
    12: "jayas_greeting",
    13: "justiciars_portal",
    14: "leafkin_druid",
    15: "mask_of_immolation",
    16: "ob_nixilis_the_hate-twisted",
    17: "plains",
    18: "plaguecrafter",
    19: "rabid_bite",
    20: "risen_reef",
    21: "roaming_throne",
    22: "soldier",
    23: "tenth_district_guard",
    24: "three_tree_city",
    25: "undead_servant",
    26: "unsummon",
    27: "vorstclaw",
    28: "winged_words",
    29: "wolf",
    30: "zombie_army"
}

for f in os.listdir(input_dir):
    file_path = os.path.join(input_dir, f)
    if not f.lower().endswith(".heic"):
        continue

    img = Image.open(file_path).convert("RGB")

    w, h = img.size

    if w >= h:
        new_w = TARGET
        new_h = int(h * TARGET / w)
    else:
        new_h = TARGET
        new_w = int(w * TARGET / h)

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    temp_path = os.path.join(input_dir, "temp_preview.jpg")
    img.save(temp_path)
    win_path = subprocess.check_output(["wslpath", "-w", temp_path]).decode().strip()
    subprocess.run(["cmd.exe", "/c", "start", "", win_path])
    key = int(input("Enter output file value (without extension): "))

    out_path = os.path.join(output_dir, f"{card_map[key]}_004.jpeg")

    # small delay helps Windows release file lock
    time.sleep(0.2) 

    os.remove(temp_path)

    img.save(
        out_path,
        "JPEG",
        quality=88,
        optimize=True,
        subsampling=2
    )