"""
Zero-shot evaluation of MallaNet student checkpoint on NHCD dataset.

NHCD (Nepali Handwritten Character Dataset, Acharya et al. 2015) is used as a
proxy for CPAR because the CPAR dataset's S3 bucket (cpar.s3.amazonaws.com)
is permanently inaccessible — the dataset author confirmed their AWS account
was deactivated (see github.com/gaganmanku96/CPAR issue #1).

NHCD properties:
- Identical 36-consonant character inventory to DHCD (our training set)
- Same Devanagari script
- 28x28 grayscale JPEG images (same as CPAR format)
- Real handwritten samples (NOT synthetic)
- Independent collection from DHCD (different writers, institution, country)
- 205 samples/consonant class, 288 samples/digit class = 10260 total

Key preprocessing difference:
- NHCD: dark ink on white background (mean ~0.80 in [0,1])
- DHCD: white ink on dark background (mean ~0.24 in [0,1])
- Fix: invert NHCD images before normalization (arr = 1.0 - arr)

Source: ashokpant/devanagari-character-dataset on Kaggle
        Acharya S, Gyawali PK, Jha RK, ICACSIS 2015
"""

import sys
import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# ── model setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, '/home/cdjk/github/mallanet/src')
from devnet.models.student import DevNet

CKPT_PATH = '/home/cdjk/github/mallanet/results/student_distilled_seed0/best.pth'
print("Loading checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
cfg = ckpt['config']
model = DevNet(tuple(cfg['widths']), tuple(cfg['depths']), 46, cfg['dropout'])
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

MEAN = ckpt['mean']   # 0.24000466
STD  = ckpt['std']    # 0.38653011
print(f"  widths={cfg['widths']}, depths={cfg['depths']}, dropout={cfg['dropout']}")
print(f"  normalisation: mean={MEAN:.6f}, std={STD:.6f}")

# ── DHCD class ordering (alphabetical sort of folder names, 0-indexed) ────────
DHCD_CLASSES = [
    'character_10_yna',       # 0  ञ
    'character_11_taamatar',  # 1  त
    'character_12_thaa',      # 2  थ
    'character_13_daa',       # 3  द
    'character_14_dhaa',      # 4  ध
    'character_15_adna',      # 5  ण
    'character_16_tabala',    # 6  ट
    'character_17_tha',       # 7  ठ
    'character_18_da',        # 8  ड
    'character_19_dha',       # 9  ढ
    'character_1_ka',         # 10 क
    'character_20_na',        # 11 न
    'character_21_pa',        # 12 प
    'character_22_pha',       # 13 फ
    'character_23_ba',        # 14 ब
    'character_24_bha',       # 15 भ
    'character_25_ma',        # 16 म
    'character_26_yaw',       # 17 य
    'character_27_ra',        # 18 र
    'character_28_la',        # 19 ल
    'character_29_waw',       # 20 व
    'character_2_kha',        # 21 ख
    'character_30_motosaw',   # 22 श
    'character_31_petchiryakha', # 23 ष
    'character_32_patalosaw', # 24 स
    'character_33_ha',        # 25 ह
    'character_34_chhya',     # 26 क्ष
    'character_35_tra',       # 27 त्र
    'character_36_gya',       # 28 ज्ञ
    'character_3_ga',         # 29 ग
    'character_4_gha',        # 30 घ
    'character_5_kna',        # 31 ङ
    'character_6_cha',        # 32 च
    'character_7_chha',       # 33 छ
    'character_8_ja',         # 34 ज
    'character_9_jha',        # 35 झ
    'digit_0',  # 36
    'digit_1',  # 37
    'digit_2',  # 38
    'digit_3',  # 39
    'digit_4',  # 40
    'digit_5',  # 41
    'digit_6',  # 42
    'digit_7',  # 43
    'digit_8',  # 44
    'digit_9',  # 45
]

# ── NHCD → DHCD index mapping ─────────────────────────────────────────────────
NHCD_CONSONANT_TO_DHCD = {
     1: DHCD_CLASSES.index('character_1_ka'),
     2: DHCD_CLASSES.index('character_2_kha'),
     3: DHCD_CLASSES.index('character_3_ga'),
     4: DHCD_CLASSES.index('character_4_gha'),
     5: DHCD_CLASSES.index('character_5_kna'),
     6: DHCD_CLASSES.index('character_6_cha'),
     7: DHCD_CLASSES.index('character_7_chha'),
     8: DHCD_CLASSES.index('character_8_ja'),
     9: DHCD_CLASSES.index('character_9_jha'),
    10: DHCD_CLASSES.index('character_10_yna'),
    11: DHCD_CLASSES.index('character_16_tabala'),   # ट
    12: DHCD_CLASSES.index('character_17_tha'),      # ठ
    13: DHCD_CLASSES.index('character_18_da'),       # ड
    14: DHCD_CLASSES.index('character_19_dha'),      # ढ
    15: DHCD_CLASSES.index('character_15_adna'),     # ण
    16: DHCD_CLASSES.index('character_11_taamatar'), # त
    17: DHCD_CLASSES.index('character_12_thaa'),     # थ
    18: DHCD_CLASSES.index('character_13_daa'),      # द
    19: DHCD_CLASSES.index('character_14_dhaa'),     # ध
    20: DHCD_CLASSES.index('character_20_na'),
    21: DHCD_CLASSES.index('character_21_pa'),
    22: DHCD_CLASSES.index('character_22_pha'),
    23: DHCD_CLASSES.index('character_23_ba'),
    24: DHCD_CLASSES.index('character_24_bha'),
    25: DHCD_CLASSES.index('character_25_ma'),
    26: DHCD_CLASSES.index('character_26_yaw'),
    27: DHCD_CLASSES.index('character_27_ra'),
    28: DHCD_CLASSES.index('character_28_la'),
    29: DHCD_CLASSES.index('character_29_waw'),
    30: DHCD_CLASSES.index('character_30_motosaw'),
    31: DHCD_CLASSES.index('character_31_petchiryakha'),
    32: DHCD_CLASSES.index('character_32_patalosaw'),
    33: DHCD_CLASSES.index('character_33_ha'),
    34: DHCD_CLASSES.index('character_34_chhya'),
    35: DHCD_CLASSES.index('character_35_tra'),
    36: DHCD_CLASSES.index('character_36_gya'),
}

NHCD_DIGIT_TO_DHCD = {
    d: DHCD_CLASSES.index(f'digit_{d}') for d in range(10)
}

# ── load NHCD images with inversion ───────────────────────────────────────────
NHCD_ROOT = Path('/tmp/cpar_eval/nhcd_data/nhcd/nhcd')
CONSONANT_DIR = NHCD_ROOT / 'consonants'
NUMERAL_DIR   = NHCD_ROOT / 'numerals'

def load_nhcd_images_resized(root_dir, label_map, mean, std, target_size=32):
    """Load all images, invert polarity, resize to target_size, normalize."""
    images = []
    labels = []
    for nhcd_lbl, dhcd_lbl in sorted(label_map.items()):
        class_dir = root_dir / str(nhcd_lbl)
        if not class_dir.exists():
            print(f"  WARNING: {class_dir} not found")
            continue
        for img_path in sorted(class_dir.glob('*.jpg')):
            img = Image.open(img_path).convert('L')
            img = img.resize((target_size, target_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = 1.0 - arr   # INVERT: NHCD is dark-on-white, DHCD is white-on-dark
            arr = (arr - mean) / std
            images.append(arr)
            labels.append(dhcd_lbl)
    return np.array(images), np.array(labels)

print("\nLoading NHCD consonant images (with polarity inversion)...")
x_con, y_con = load_nhcd_images_resized(CONSONANT_DIR, NHCD_CONSONANT_TO_DHCD, MEAN, STD)
print(f"  {len(x_con)} images, {len(np.unique(y_con))} classes")

print("Loading NHCD numeral images (with polarity inversion)...")
x_num, y_num = load_nhcd_images_resized(NUMERAL_DIR, NHCD_DIGIT_TO_DHCD, MEAN, STD)
print(f"  {len(x_num)} images, {len(np.unique(y_num))} classes")

x_all = np.concatenate([x_con, x_num], axis=0)
y_all = np.concatenate([y_con, y_num], axis=0)
print(f"\nTotal: {len(x_all)} images across {len(np.unique(y_all))} overlapping classes")

# ── batch inference ────────────────────────────────────────────────────────────
x_tensor = torch.tensor(x_all, dtype=torch.float32).unsqueeze(1)  # (N,1,32,32)

BATCH_SIZE = 256
all_preds = []
print(f"Running inference (batch_size={BATCH_SIZE})...")
with torch.no_grad():
    for i in range(0, len(x_tensor), BATCH_SIZE):
        batch = x_tensor[i:i+BATCH_SIZE]
        logits = model(batch)
        all_preds.extend(logits.argmax(dim=1).numpy().tolist())

all_preds = np.array(all_preds)

# ── compute accuracy ───────────────────────────────────────────────────────────
correct = (all_preds == y_all).sum()
total   = len(y_all)
acc     = correct / total * 100.0

print(f"\n{'='*60}")
print(f"Dataset:              NHCD (proxy for CPAR; CPAR S3 dead)")
print(f"N overlapping classes:{len(np.unique(y_all))}/46")
print(f"N images evaluated:   {total}")
print(f"Zero-shot accuracy:   {acc:.2f}%")
print(f"{'='*60}")

# ── per-class breakdown ────────────────────────────────────────────────────────
print("\nPer-class accuracy:")
unique_classes = np.unique(y_all)
class_accs = {}
for c in unique_classes:
    mask = (y_all == c)
    c_correct = (all_preds[mask] == c).sum()
    c_total   = mask.sum()
    c_acc     = c_correct / c_total * 100.0
    class_accs[c] = (c_acc, c_total, DHCD_CLASSES[c])
    print(f"  [{c:2d}] {DHCD_CLASSES[c]:35s}  {c_acc:6.1f}%  ({c_correct}/{c_total})")

sorted_classes = sorted(class_accs.items(), key=lambda x: x[1][0])
print("\n5 worst classes:")
for c, (acc_c, n, name) in sorted_classes[:5]:
    print(f"  {name:35s}  {acc_c:.1f}%")
print("\n5 best classes:")
for c, (acc_c, n, name) in sorted_classes[-5:]:
    print(f"  {name:35s}  {acc_c:.1f}%")

con_mask = (y_all < 36)
num_mask = (y_all >= 36)
con_acc = (all_preds[con_mask] == y_all[con_mask]).mean() * 100
num_acc = (all_preds[num_mask] == y_all[num_mask]).mean() * 100
print(f"\nConsonants only: {con_acc:.2f}%  ({con_mask.sum()} images)")
print(f"Digits only:     {num_acc:.2f}%  ({num_mask.sum()} images)")
print(f"\n(Chance baseline: {100/46:.2f}% for 46-class uniform)")
print("Done.")
