"""
Zero-shot evaluation of DevNet (DHCD-trained) on Prashanth et al. 2021 Devanagari digits.
Dataset: Mendeley pxrnvp4yy8 v4, digit classes only.
"""
import sys, os, glob, json
sys.path.insert(0, '/home/cdjk/github/mallanet/src')

import torch
import numpy as np
from PIL import Image
from pathlib import Path

# --- Load model ---
print("Loading model...")
ckpt = torch.load(
    '/home/cdjk/github/mallanet/results/student_distilled_seed0/best.pth',
    map_location='cpu',
    weights_only=False
)
cfg = ckpt['config']
from devnet.models.student import DevNet
model = DevNet(tuple(cfg['widths']), tuple(cfg['depths']), 46, cfg['dropout'])
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Model loaded: widths={cfg['widths']}, depths={cfg['depths']}, dropout={cfg['dropout']}, classes=46")

# --- Dataset paths ---
DATA_ROOT = '/tmp/prashanth_eval/dataset1/Dataset_1/Numbers'
MEAN = 0.24000466
STD = 0.38653011

# DHCD digit class mapping: digit_N -> class index 36+N
DIGIT_TO_CLASS = {str(d): 36 + d for d in range(10)}
print(f"Digit-to-class mapping: {DIGIT_TO_CLASS}")

# --- Check polarity by sampling a few images ---
sample_imgs = glob.glob(os.path.join(DATA_ROOT, '0', '*.jpg'))[:5]
sample_means = []
for p in sample_imgs:
    arr = np.array(Image.open(p).convert('L'), dtype=np.float32) / 255.0
    sample_means.append(arr.mean())
dataset_mean = sum(sample_means) / len(sample_means)
print(f"\nDataset image mean (sample): {dataset_mean:.4f}")
print(f"DHCD training mean: {MEAN:.4f}")
# DHCD: black background (mean ~0), white strokes → overall mean ~0.24
# This dataset: white background (mean ~0.97), black strokes → need inversion
INVERT = dataset_mean > 0.5
print(f"Polarity inversion needed: {INVERT}")

# --- Evaluate ---
correct = 0
total = 0
per_class_correct = {str(d): 0 for d in range(10)}
per_class_total = {str(d): 0 for d in range(10)}
errors_by_true_pred = {}

print("\nEvaluating...")
with torch.no_grad():
    for digit in sorted(os.listdir(DATA_ROOT)):
        digit_dir = os.path.join(DATA_ROOT, digit)
        if not os.path.isdir(digit_dir) or digit not in DIGIT_TO_CLASS:
            continue
        
        target_class = DIGIT_TO_CLASS[digit]
        imgs = glob.glob(os.path.join(digit_dir, '*.jpg'))
        
        for img_path in imgs:
            # Preprocess
            img = Image.open(img_path).convert('L').resize((32, 32), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            if INVERT:
                arr = 1.0 - arr
            arr = (arr - MEAN) / STD
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, 32, 32]
            
            # Forward pass
            logits = model(tensor)  # [1, 46]
            pred_class = logits.argmax(dim=1).item()
            
            # Score
            per_class_total[digit] += 1
            total += 1
            if pred_class == target_class:
                correct += 1
                per_class_correct[digit] += 1
            else:
                key = (target_class, pred_class)
                errors_by_true_pred[key] = errors_by_true_pred.get(key, 0) + 1

# --- Report ---
accuracy = correct / total * 100 if total > 0 else 0
print(f"\n{'='*60}")
print(f"Dataset: Prashanth et al. 2021 (Mendeley pxrnvp4yy8 v4 - digits)")
print(f"N digit classes: 10")
print(f"N images evaluated: {total}")
print(f"Correct: {correct}")
print(f"Zero-shot accuracy: {accuracy:.2f}%")
print(f"Polarity inverted: {'yes' if INVERT else 'no'}")
print(f"{'='*60}")

print("\nPer-class accuracy:")
for digit in sorted(per_class_correct.keys()):
    c = per_class_correct[digit]
    t = per_class_total[digit]
    pct = c/t*100 if t > 0 else 0
    print(f"  Digit {digit} (DHCD class {DIGIT_TO_CLASS[digit]}): {c}/{t} = {pct:.1f}%")

print("\nTop confusion pairs:")
top_errors = sorted(errors_by_true_pred.items(), key=lambda x: -x[1])[:10]
for (true_c, pred_c), cnt in top_errors:
    # Convert class idx back to digit label
    true_digit = true_c - 36 if true_c >= 36 else true_c
    pred_digit = pred_c - 36 if pred_c >= 36 else pred_c
    print(f"  True digit_{true_digit} (class {true_c}) -> Predicted class {pred_c} (digit_{pred_digit} if digit): {cnt}")

# Save result
result = {
    "dataset": "Prashanth et al. 2021 (Mendeley pxrnvp4yy8 v4)",
    "n_digit_classes": 10,
    "n_images": total,
    "correct": correct,
    "accuracy_pct": round(accuracy, 4),
    "polarity_inverted": INVERT,
    "dataset_image_mean": round(dataset_mean, 4),
    "per_class_accuracy": {d: round(per_class_correct[d]/per_class_total[d]*100, 2) for d in per_class_correct},
}
with open('/tmp/prashanth_eval/results.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to /tmp/prashanth_eval/results.json")
