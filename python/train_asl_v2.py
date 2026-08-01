"""
Stage 1 (ASL, v2): Train the float32 reference MLP with normalization,
augmentation, and regularization.

    784 -> 128 -> 24

CHANGES FROM v1 AND WHY THEY DON'T COST HARDWARE
------------------------------------------------
1. Global input normalization (x - MEAN) / STD.
   Normalization is LINEAR, so at export time it folds into layer 1:
       W((x-mu)/sigma) + b  ==  (W/sigma)x + (b - W*mu/sigma)
   Bake mu/sigma into the exported weights and the FPGA still receives a
   raw uint8 pixel and performs identical MACs. Zero added hardware.
   (Same trick production toolchains use to fold batchnorm into conv.)

2. Data augmentation -- training only. Inference path unchanged.

3. Dropout -- identity at eval time. Inference path unchanged.

4. Cosine LR schedule + more epochs -- training only.

5. Hidden 64 -> 128. This one DOES cost hardware:
       784*128 + 128*24 = 103,424 weights = 827 Kbit @ INT8
       = 46% of the Artix-7 35T's 1,800 Kbit BRAM. Fits.
       Layer-1 MACs 50,176 -> 100,352; at N=16 that is 6,272 cycles,
       ~63 us/inference at 100 MHz. Still comfortable.

Dataset: Sign Language MNIST (Kaggle: datamunge/sign-language-mnist)
Place sign_mnist_train.csv and sign_mnist_test.csv in ./data/
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

torch.manual_seed(0)

HIDDEN = 128
EPOCHS = 60
BATCH = 128
LR = 2e-3
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4

TRAIN_CSV = "./data/sign_mnist_train.csv"
TEST_CSV = "./data/sign_mnist_test.csv"

# 9 (J) and 25 (Z) require motion and never appear. Remap to contiguous 0..23
# so the output layer carries no permanently-dead neurons.
RAW_LABELS = [i for i in range(26) if i not in (9, 25)]
RAW_TO_IDX = {raw: i for i, raw in enumerate(RAW_LABELS)}
IDX_TO_LETTER = [chr(ord("A") + raw) for raw in RAW_LABELS]
NUM_CLASSES = len(RAW_LABELS)  # 24

# Letters that are near-identical at 28x28: all closed fists (M/N/S/T) or
# two-fingers-up (K/V/U). Tracked separately because residual error here is
# largely a property of the data, not the model.
FIST_FAMILY = ["M", "N", "S", "T"]
TWO_FINGER = ["K", "U", "V"]


class MLP(nn.Module):
    def __init__(self, hidden=HIDDEN, n_out=NUM_CLASSES, p_drop=DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, n_out)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)


class SignDataset(Dataset):
    """Holds normalized (1,28,28) tensors; augments on the fly when training.

    Augmentation happens AFTER normalization so the affine fill value of 0
    corresponds to the dataset mean rather than to black.
    """

    def __init__(self, images, labels, augment):
        self.images = images
        self.labels = labels
        self.augment = augment
        self.aug = transforms.RandomAffine(
            degrees=12,
            translate=(0.12, 0.12),
            scale=(0.88, 1.12),
            shear=6,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i]
        if self.augment:
            x = self.aug(x)
        return x.reshape(-1), self.labels[i]


def load_csv(path):
    """CSV layout: col 0 = label, cols 1..784 = pixel values 0-255."""
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int32)
    labels_raw = raw[:, 0]
    pixels = raw[:, 1:].astype(np.float32) / 255.0

    unknown = set(labels_raw.tolist()) - set(RAW_TO_IDX)
    if unknown:
        raise ValueError(f"unexpected labels in {path}: {sorted(unknown)}")

    labels = np.array([RAW_TO_IDX[v] for v in labels_raw], dtype=np.int64)
    images = torch.from_numpy(pixels).reshape(-1, 1, 28, 28)
    return images, torch.from_numpy(labels)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def per_class_accuracy(model, loader):
    model.eval()
    hit = np.zeros(NUM_CLASSES, dtype=np.int64)
    seen = np.zeros(NUM_CLASSES, dtype=np.int64)
    for x, y in loader:
        pred = model(x).argmax(dim=1)
        for cls in range(NUM_CLASSES):
            mask = y == cls
            seen[cls] += mask.sum().item()
            hit[cls] += (pred[mask] == cls).sum().item()
    return hit, seen


def group_accuracy(hit, seen, letters):
    idx = [IDX_TO_LETTER.index(l) for l in letters if l in IDX_TO_LETTER]
    h, s = hit[idx].sum(), seen[idx].sum()
    return 100.0 * h / s if s else float("nan")


def main():
    train_img, train_lab = load_csv(TRAIN_CSV)
    test_img, test_lab = load_csv(TEST_CSV)

    # Normalization constants come from the TRAINING set only -- computing
    # them over test data would leak. Scalars, not per-pixel: a single
    # mu/sigma pair folds cleanly into layer 1 at export time.
    MEAN = train_img.mean().item()
    STD = train_img.std().item()
    print(f"train {len(train_lab)} samples, test {len(test_lab)} samples, "
          f"{NUM_CLASSES} classes")
    print(f"normalization  MEAN={MEAN:.6f}  STD={STD:.6f}   "
          f"<-- fold these into fc1 at export")

    train_img = (train_img - MEAN) / STD
    test_img = (test_img - MEAN) / STD

    train_loader = DataLoader(
        SignDataset(train_img, train_lab, augment=True),
        batch_size=BATCH, shuffle=True, num_workers=2)
    test_loader = DataLoader(
        SignDataset(test_img, test_lab, augment=False), batch_size=1000)

    model = MLP()
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossfn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            lossfn(model(x), y).backward()
            opt.step()
        sched.step()
        acc = evaluate(model, test_loader)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "mlp_asl_fp32.pt")
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  test acc {acc*100:.2f}%   "
                  f"lr {sched.get_last_lr()[0]:.2e}")

    print(f"\nFP32 baseline accuracy: {best*100:.2f}%   <-- record this number")
    print("saved mlp_asl_fp32.pt (best epoch)")

    model.load_state_dict(torch.load("mlp_asl_fp32.pt"))

    print("\nweight ranges -- these set the per-layer quantization scales:")
    for name, p in model.named_parameters():
        print(f"  {name:12s} shape {tuple(p.shape)}  "
              f"min {p.min():+.4f}  max {p.max():+.4f}")

    hit, seen = per_class_accuracy(model, test_loader)
    print("\nper-class accuracy:")
    for cls in range(NUM_CLASSES):
        pct = 100.0 * hit[cls] / seen[cls] if seen[cls] else float("nan")
        flag = ""
        if IDX_TO_LETTER[cls] in FIST_FAMILY:
            flag = "  <- fist family"
        elif IDX_TO_LETTER[cls] in TWO_FINGER:
            flag = "  <- two-finger family"
        print(f"  {cls:2d}  {IDX_TO_LETTER[cls]}  {pct:6.2f}%  "
              f"(n={seen[cls]}){flag}")

    print(f"\nfist family (M/N/S/T):    {group_accuracy(hit, seen, FIST_FAMILY):.2f}%")
    print(f"two-finger (K/U/V):       {group_accuracy(hit, seen, TWO_FINGER):.2f}%")
    others = [l for l in IDX_TO_LETTER
              if l not in FIST_FAMILY and l not in TWO_FINGER]
    print(f"all other letters:        {group_accuracy(hit, seen, others):.2f}%")


if __name__ == "__main__":
    main()
