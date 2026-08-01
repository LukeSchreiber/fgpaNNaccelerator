"""
Stage 2: Fold normalization, quantize to int8, export .mem files.

PIPELINE THE HARDWARE WILL IMPLEMENT
------------------------------------
    raw pixel p (uint8, 0..255)
      -> acc1 = sum(qW1 * p) + qb1              (int32)
      -> a1   = clip(acc1 >> SHIFT1, 0, 127)    (uint8, ReLU folded in)
      -> acc2 = sum(qW2 * a1) + qb2             (int32)
      -> argmax(acc2)                           (class index)

No floats. No division. No preprocessing. Three design points behind that:

1. FOLDING. Training fed fc1 the value (p/255 - MEAN)/STD. That is linear,
   so it collapses into the layer-1 parameters:

       W*((p/255 - MEAN)/STD) + b
         = (W/(255*STD)) * p  +  (b - (MEAN/STD)*rowsum(W))

   Fold once here, and the FPGA receives a raw pixel and does ordinary MACs.
   Zero added hardware. Same trick as folding batchnorm into conv weights.

2. BIASES AT ACCUMULATOR SCALE, int32. After training, biases were 3-5x
   larger in magnitude than weights. Quantizing both with one int8 scale
   would clip the biases badly. Instead biases are stored int32 at the
   accumulator's own scale (input_scale * weight_scale), so they add in
   directly with no shifting. Costs nothing: 152 biases vs 103,424 weights.

3. POWER-OF-TWO REQUANTIZATION. Exact requantization multiplies the
   accumulator by a fractional scale -- a multiplier plus a rounding unit
   per layer. A right-shift is nearly free. Slight accuracy cost, measured
   below, reported in the results table.

No final requantization is needed: argmax is invariant under a positive
scale factor, so the hardware compares raw int32 accumulators.
"""

import numpy as np
import torch
import torch.nn as nn

# Must match training. Printed by train_asl_v2.py.
MEAN = 0.624671
STD = 0.191253

HIDDEN = 128
W_BITS = 8                       # sweep this later: 4 / 8 / 16
A_BITS = 8                       # activation width between layers

TRAIN_CSV = "./data/sign_mnist_train.csv"
TEST_CSV = "./data/sign_mnist_test.csv"
CKPT = "mlp_asl_fp32.pt"
MEM_DIR = "./mem"

RAW_LABELS = [i for i in range(26) if i not in (9, 25)]
RAW_TO_IDX = {raw: i for i, raw in enumerate(RAW_LABELS)}
IDX_TO_LETTER = [chr(ord("A") + raw) for raw in RAW_LABELS]
NUM_CLASSES = len(RAW_LABELS)


class MLP(nn.Module):
    def __init__(self, hidden=HIDDEN, n_out=NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, n_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.view(-1, 784))))


def load_csv(path):
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int32)
    labels = np.array([RAW_TO_IDX[v] for v in raw[:, 0]], dtype=np.int64)
    pixels = raw[:, 1:].astype(np.int32)          # keep as raw 0..255
    return pixels, labels


def fold_normalization(W1, b1):
    """Collapse the /255 rescale and (x-MEAN)/STD into layer-1 parameters."""
    W1f = W1 / (255.0 * STD)
    b1f = b1 - (MEAN / STD) * W1.sum(axis=1)
    return W1f, b1f


def symmetric_scale(x, bits):
    """Largest-magnitude value maps to the top representable integer."""
    qmax = 2 ** (bits - 1) - 1                    # 127 for 8-bit
    return np.abs(x).max() / qmax, qmax


def quantize_weights(W, bits):
    scale, qmax = symmetric_scale(W, bits)
    q = np.clip(np.round(W / scale), -qmax, qmax).astype(np.int32)
    return q, scale


def pick_shift(acc, target_max, pct=99.9):
    """Smallest right-shift that keeps (almost) all activations in range.

    Calibrated on a percentile rather than the true max so one outlier
    image does not cost precision for every other image.
    """
    hi = np.percentile(acc[acc > 0], pct) if (acc > 0).any() else 1.0
    shift = 0
    while hi / (2 ** shift) > target_max:
        shift += 1
    return shift


def forward_int(px, qW1, qb1, shift1, qW2, qb2):
    """Integer-only forward pass. This is the golden model the RTL must match.

    Every operation here has a direct hardware counterpart:
      @        -> MAC array
      +        -> accumulator preload
      >>       -> barrel shifter
      clip     -> saturating register
      argmax   -> comparator tree
    """
    acc1 = px @ qW1.T + qb1                       # int32
    a1 = np.clip(acc1 >> shift1, 0, 2 ** A_BITS - 1)   # ReLU + requantize
    acc2 = a1 @ qW2.T + qb2                       # int32
    return acc2


def write_mem(path, values, bits):
    """One hex word per line, two's complement, width = bits."""
    mask = (1 << bits) - 1
    with open(path, "w") as f:
        for v in np.asarray(values).reshape(-1):
            f.write(f"{int(v) & mask:0{(bits + 3) // 4}x}\n")


def main():
    import os
    os.makedirs(MEM_DIR, exist_ok=True)

    model = MLP()
    model.load_state_dict(torch.load(CKPT))
    model.eval()

    W1 = model.fc1.weight.detach().numpy().astype(np.float64)
    b1 = model.fc1.bias.detach().numpy().astype(np.float64)
    W2 = model.fc2.weight.detach().numpy().astype(np.float64)
    b2 = model.fc2.bias.detach().numpy().astype(np.float64)

    print("pre-fold ranges:")
    print(f"  W1 [{W1.min():+.4f}, {W1.max():+.4f}]   "
          f"b1 [{b1.min():+.4f}, {b1.max():+.4f}]")

    W1f, b1f = fold_normalization(W1, b1)
    print("post-fold ranges (scales MUST be computed from these):")
    print(f"  W1 [{W1f.min():+.6f}, {W1f.max():+.6f}]   "
          f"b1 [{b1f.min():+.4f}, {b1f.max():+.4f}]")

    # --- layer 1 -----------------------------------------------------------
    # Input scale is 1.0: the hardware consumes the raw integer pixel, so the
    # accumulator's scale is exactly the weight scale.
    qW1, s_w1 = quantize_weights(W1f, W_BITS)
    qb1 = np.round(b1f / s_w1).astype(np.int32)

    # --- calibrate the layer-1 shift on training data -----------------------
    train_px, train_y = load_csv(TRAIN_CSV)
    calib = train_px[:2000]
    acc1_calib = calib @ qW1.T + qb1
    shift1 = pick_shift(acc1_calib, 2 ** A_BITS - 1)
    s_a1 = s_w1 * (2 ** shift1)                  # real value per a1 unit
    print(f"\nlayer 1: weight scale {s_w1:.3e}   shift {shift1}   "
          f"activation scale {s_a1:.3e}")

    clipped = (np.clip(acc1_calib >> shift1, 0, None) > 2 ** A_BITS - 1).mean()
    print(f"         activations saturating at {clipped*100:.3f}% "
          f"(low is good; nonzero is fine)")

    # --- layer 2 -----------------------------------------------------------
    qW2, s_w2 = quantize_weights(W2, W_BITS)
    qb2 = np.round(b2 / (s_w2 * s_a1)).astype(np.int32)
    print(f"layer 2: weight scale {s_w2:.3e}")

    # --- accumulator width check -------------------------------------------
    peak = np.abs(calib @ qW1.T + qb1).max()
    print(f"\npeak |acc1| observed: {peak:,}  -> needs "
          f"{int(np.ceil(np.log2(peak))) + 1} bits (int32 chosen)")

    # --- evaluate ----------------------------------------------------------
    test_px, test_y = load_csv(TEST_CSV)

    with torch.no_grad():
        xf = torch.from_numpy(
            ((test_px / 255.0 - MEAN) / STD).astype(np.float32))
        fp32_pred = model(xf).argmax(dim=1).numpy()
    fp32_acc = (fp32_pred == test_y).mean()

    int_pred = forward_int(test_px, qW1, qb1, shift1, qW2, qb2).argmax(axis=1)
    int_acc = (int_pred == test_y).mean()
    agree = (int_pred == fp32_pred).mean()

    print(f"\nFP32 accuracy      {fp32_acc*100:.2f}%")
    print(f"INT{W_BITS} accuracy      {int_acc*100:.2f}%")
    print(f"drop               {(fp32_acc - int_acc)*100:+.2f} points")
    print(f"prediction agreement with FP32: {agree*100:.2f}%")

    # --- export ------------------------------------------------------------
    write_mem(f"{MEM_DIR}/w1.mem", qW1, W_BITS)
    write_mem(f"{MEM_DIR}/b1.mem", qb1, 32)
    write_mem(f"{MEM_DIR}/w2.mem", qW2, W_BITS)
    write_mem(f"{MEM_DIR}/b2.mem", qb2, 32)

    # A few test images for the on-board demo, plus their expected answers.
    n_demo = 16
    write_mem(f"{MEM_DIR}/images.mem", test_px[:n_demo], 8)
    write_mem(f"{MEM_DIR}/labels.mem", test_y[:n_demo], 8)

    np.savez(f"{MEM_DIR}/params.npz",
             qW1=qW1, qb1=qb1, qW2=qW2, qb2=qb2,
             shift1=shift1, s_w1=s_w1, s_a1=s_a1, s_w2=s_w2)

    print(f"\nwrote {MEM_DIR}/: w1 w2 b1 b2 images labels + params.npz")
    print(f"  w1.mem  {qW1.size:,} words   w2.mem  {qW2.size:,} words")
    print(f"\nRTL constants:")
    print(f"  SHIFT1     = {shift1}")
    print(f"  HIDDEN     = {HIDDEN}")
    print(f"  NUM_CLASS  = {NUM_CLASSES}")


if __name__ == "__main__":
    main()
