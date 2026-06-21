"""
Evaluate MallaNet baseline zero-shot on NHCD dataset.
NHCD: Pant 2012, Kaggle ashokpant/devanagari-character-dataset
"""
import sys, torch, numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

sys.path.insert(0, 'src')
from devnet.models.mallanet_baseline import EnhancedBMCNNwHFCs
from devnet.train import resolve_device

# Load permutation mapping recovered from DHCD test set
permutation = np.load('/tmp/mallanet_ood/permutation.npy')  # model_output_idx -> dhcd_idx

# NHCD consonant folder index (1-36) -> DHCD class index
# Based on labels.csv and DHCD ordering:
# DHCD consonant indices 0-35:
# 0=yna(ञ), 1=taamatar(त), 2=thaa(थ), 3=daa(द), 4=dhaa(ध), 5=adna(न_alt/ण),
# 6=tabala(ट), 7=tha(ठ), 8=da(ड), 9=dha(ढ), 10=ka(क), 11=na(न),
# 12=pa(प), 13=pha(फ), 14=ba(ब), 15=bha(भ), 16=ma(म), 17=yaw(य),
# 18=ra(र), 19=la(ल), 20=waw(व), 21=kha(ख), 22=motosaw(श), 23=petchiryakha(ष),
# 24=patalosaw(स), 25=ha(ह), 26=chhya(क्ष), 27=tra(त्र), 28=gya(ज्ञ),
# 29=ga(ग), 30=gha(घ), 31=kna(ङ), 32=cha(च), 33=chha(छ), 34=ja(ज), 35=jha(झ)
#
# NHCD labels.csv consonants (folder_num -> character):
# 1=क, 2=ख, 3=ग, 4=घ, 5=ङ, 6=च, 7=छ, 8=ज, 9=झ, 10=ञ,
# 11=ट, 12=ठ, 13=ड, 14=ढ, 15=ण, 16=त, 17=थ, 18=द, 19=ध, 20=न,
# 21=प, 22=फ, 23=ब, 24=भ, 25=म, 26=य, 27=र, 28=ल, 29=व, 30=श,
# 31=ष, 32=स, 33=ह, 34=क्ष, 35=त्र, 36=ज्ञ

# nhcd_consonant_folder -> dhcd_index
nhcd_consonant_to_dhcd = {
    1:  10,  # क -> ka
    2:  21,  # ख -> kha
    3:  29,  # ग -> ga
    4:  30,  # घ -> gha
    5:  31,  # ङ -> kna
    6:  32,  # च -> cha
    7:  33,  # छ -> chha
    8:  34,  # ज -> ja
    9:  35,  # झ -> jha
    10:  0,  # ञ -> yna
    11:  6,  # ट -> tabala
    12:  7,  # ठ -> tha
    13:  8,  # ड -> da
    14:  9,  # ढ -> dha
    15:  5,  # ण -> adna (note: DHCD index 5 is ण in DHCD)
    16:  1,  # त -> taamatar
    17:  2,  # थ -> thaa
    18:  3,  # द -> daa
    19:  4,  # ध -> dhaa
    20: 11,  # न -> na
    21: 12,  # प -> pa
    22: 13,  # फ -> pha
    23: 14,  # ब -> ba
    24: 15,  # भ -> bha
    25: 16,  # म -> ma
    26: 17,  # य -> yaw
    27: 18,  # र -> ra
    28: 19,  # ल -> la
    29: 20,  # व -> waw
    30: 22,  # श -> motosaw
    31: 23,  # ष -> petchiryakha
    32: 24,  # स -> patalosaw
    33: 25,  # ह -> ha
    34: 26,  # क्ष -> chhya
    35: 27,  # त्र -> tra
    36: 28,  # ज्ञ -> gya
}

# nhcd_numeral_folder (0-9) -> dhcd_index (36-45)
nhcd_numeral_to_dhcd = {i: 36 + i for i in range(10)}

NHCD_ROOT = Path('/tmp/mallanet_ood/nhcd/nhcd/nhcd')


class NHCDDataset(Dataset):
    def __init__(self, transform=None):
        self.samples = []  # (path, dhcd_label)
        self.transform = transform
        # consonants
        for folder_num, dhcd_idx in nhcd_consonant_to_dhcd.items():
            folder = NHCD_ROOT / 'consonants' / str(folder_num)
            for img_path in sorted(folder.glob('*.jpg')):
                self.samples.append((img_path, dhcd_idx))
        # numerals
        for folder_num, dhcd_idx in nhcd_numeral_to_dhcd.items():
            folder = NHCD_ROOT / 'numerals' / str(folder_num)
            for img_path in sorted(folder.glob('*.jpg')):
                self.samples.append((img_path, dhcd_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L')
        # Polarity: white background, dark ink -> invert to match DHCD (black bg, white ink)
        arr = 1.0 - np.array(img, dtype=np.float32) / 255.0
        # Resize to 32x32
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
    dataset = NHCDDataset(transform=transform)
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
    overall_acc = (preds == labels).mean()

    # Consonant accuracy (dhcd indices 0-35)
    cons_mask = labels < 36
    cons_acc = (preds[cons_mask] == labels[cons_mask]).mean() if cons_mask.any() else 0.0

    # Digit accuracy (dhcd indices 36-45)
    digit_mask = labels >= 36
    digit_acc = (preds[digit_mask] == labels[digit_mask]).mean() if digit_mask.any() else 0.0

    # Count covered classes
    unique_labels = len(np.unique(labels))

    print(f"N images: {n_images}")
    print(f"N classes: {unique_labels}/46")
    print(f"Overall zero-shot accuracy: {overall_acc*100:.2f}%")
    print(f"Consonants only: {cons_acc*100:.2f}%")
    print(f"Digits only: {digit_acc*100:.2f}%")
    print(f"Polarity inversion: yes (dark-ink-on-white -> inverted)")

    # Per-class breakdown (for debugging)
    print("\nPer-class accuracy:")
    for dhcd_idx in range(46):
        mask = labels == dhcd_idx
        if mask.any():
            acc = (preds[mask] == labels[mask]).mean()
            n = mask.sum()
            print(f"  class {dhcd_idx}: {acc:.3f} ({n} images)")


if __name__ == '__main__':
    main()
