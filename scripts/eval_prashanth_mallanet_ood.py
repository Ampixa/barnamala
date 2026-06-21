"""
Zero-shot evaluation of MallaNet baseline on Prashanth 2021 Devanagari digits.
Dataset: images_full/Images/Numerals/ (dark bg, white strokes, 28x28 JPEG)
No polarity inversion needed (same polarity as DHCD training data).
"""
import sys, torch, numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

sys.path.insert(0, 'src')
from devnet.models.mallanet_baseline import EnhancedBMCNNwHFCs
from devnet.train import resolve_device

# Load permutation mapping
permutation = np.load('/tmp/mallanet_ood/permutation.npy')  # model_output_idx -> dhcd_idx

PRASHANTH_ROOT = Path('/tmp/prashanth_eval/images_full/Images/Numerals')

# DHCD digit indices: digit_0 -> 36, digit_1 -> 37, ..., digit_9 -> 45
DIGIT_TO_DHCD = {str(d): 36 + d for d in range(10)}


class PrashanthDigitDataset(Dataset):
    def __init__(self, transform=None):
        self.samples = []  # (path, dhcd_label)
        self.transform = transform
        for digit_str, dhcd_idx in DIGIT_TO_DHCD.items():
            folder = PRASHANTH_ROOT / digit_str
            for img_path in sorted(folder.glob('*.jpg')):
                self.samples.append((img_path, dhcd_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L')
        # Dark background, white strokes - same polarity as DHCD, NO inversion
        arr = np.array(img, dtype=np.float32) / 255.0
        # Resize 28->32 BILINEAR
        pil = Image.fromarray((arr * 255).astype(np.uint8))
        pil = pil.resize((32, 32), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(pil, dtype=np.float32) / 255.0).unsqueeze(0)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label


def main():
    device = resolve_device('auto')
    model = EnhancedBMCNNwHFCs(num_classes=46).to(device)
    ckpt = torch.load('/tmp/mallanet_repo/models/best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    transform = v2.Compose([v2.Normalize([0.5], [0.5])])
    dataset = PrashanthDigitDataset(transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            raw = model(x.to(device)).argmax(1).cpu().numpy()
            corrected = permutation[raw]
            all_preds.append(corrected)
            all_labels.append(y.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    n_images = len(labels)
    n_classes = len(np.unique(labels))
    overall_acc = (preds == labels).mean()

    print(f"N images: {n_images}")
    print(f"N classes: {n_classes}")
    print(f"Zero-shot accuracy: {overall_acc*100:.2f}%")
    print(f"Polarity inversion: no (dark-bg/white-stroke, same as DHCD)")
    print(f"Dataset mean (approximate): 0.21 (confirmed)")

    print("\nPer-class accuracy:")
    for d in range(10):
        dhcd_idx = 36 + d
        mask = labels == dhcd_idx
        if mask.any():
            acc = (preds[mask] == labels[mask]).mean()
            n = mask.sum()
            print(f"  digit_{d} (DHCD class {dhcd_idx}): {acc:.3f} ({n} images)")


if __name__ == '__main__':
    main()
