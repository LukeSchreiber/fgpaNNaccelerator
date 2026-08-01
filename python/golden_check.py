"""
Stage 2c: run the integer golden model on the demo images and export the
expected predictions for the RTL testbench.

The testbench compares against THESE predictions, not against the true
labels. That distinction matters: the quantized model is ~92% accurate, so
matching true labels would fail on roughly one image in twelve and tell you
nothing about whether the hardware is correct. Matching the golden model's
own output is the real claim -- bit-accurate RTL against a reference
implementation.

Also prints the expected cycle count so the testbench result can be checked
against the architectural prediction rather than just accepted.
"""

import os
import re

import numpy as np

N = 16
INPUTS = 784
MEM_DIR = "./mem"

RTL_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "rtl", "nn_accel_core.v")


def rtl_constants(path=RTL_CORE):
    """Read N/INPUTS/HIDDEN/NUM_CLASS/DRAIN_CYCLES straight out of the RTL.

    Parsed rather than duplicated. Every Python-side copy of a hardware
    constant is a copy that can go stale, and a stale cycle count is the kind
    of error that produces a confidently wrong number rather than a crash --
    the web UI would happily report a latency the hardware never had.

    Two pipeline stages sit between the MAC inputs and sum_r (mac_array's
    partial sums, then sum_r itself), so five cycles are needed to drain a
    layer; the RTL allows seven.
    """
    src = open(path).read()
    want = ("N", "INPUTS", "HIDDEN", "NUM_CLASS", "DRAIN_CYCLES")
    out = {}
    for name in want:
        m = re.search(rf"\b(?:parameter|localparam)\s+integer\s+{name}\s*=\s*(\d+)", src)
        if not m:
            raise RuntimeError(f"cannot find {name} in {path}")
        out[name] = int(m.group(1))
    return out


def expected_cycles(c=None):
    """Cycles per inference, derived from the RTL's own parameters."""
    c = c or rtl_constants()
    return (c["HIDDEN"] * (c["INPUTS"] // c["N"])
            + c["NUM_CLASS"] * (c["HIDDEN"] // c["N"])
            + 2 * (c["DRAIN_CYCLES"] + 1))

RAW_LABELS = [i for i in range(26) if i not in (9, 25)]
IDX_TO_LETTER = [chr(ord("A") + raw) for raw in RAW_LABELS]


def forward_int(px, qW1, qb1, shift1, qW2, qb2):
    """Integer-only forward pass. Must match nn_accel_core.v exactly."""
    acc1 = px @ qW1.T + qb1
    a1 = np.clip(acc1 >> shift1, 0, 255)
    acc2 = a1 @ qW2.T + qb2
    return acc1, a1, acc2


def main():
    p = np.load(f"{MEM_DIR}/params.npz")
    qW1, qb1 = p["qW1"], p["qb1"]
    qW2, qb2 = p["qW2"], p["qb2"]
    shift1 = int(p["shift1"])

    hidden = qW1.shape[0]
    n_class = qW2.shape[0]

    with open(f"{MEM_DIR}/images.mem") as f:
        px = [int(line.strip(), 16) for line in f if line.strip()]
    n_img = len(px) // INPUTS
    images = np.array(px, dtype=np.int64).reshape(n_img, INPUTS)

    with open(f"{MEM_DIR}/labels.mem") as f:
        labels = [int(line.strip(), 16) for line in f if line.strip()]

    acc1, a1, acc2 = forward_int(images, qW1, qb1, shift1, qW2, qb2)
    preds = acc2.argmax(axis=1)

    with open(f"{MEM_DIR}/preds.mem", "w") as f:
        for v in preds:
            f.write(f"{int(v):02x}\n")

    # The winning logit, two's complement, for the testbench to check as well.
    #
    # The class index alone is a weak check: it survives arithmetic that is
    # merely CLOSE. A capture strobe one cycle out of alignment, for instance,
    # reads each dot product before its last chunk of terms is added -- which
    # moved exactly one of these 16 images to a different class and left the
    # other 15 looking correct. The logit is bit-exact or it is not.
    with open(f"{MEM_DIR}/scores.mem", "w") as f:
        for i, v in enumerate(preds):
            f.write(f"{int(acc2[i, v]) & 0xFFFFFFFF:08x}\n")

    print(f"{n_img} demo images\n")
    print("  img  golden  true   match   top logit")
    for i in range(n_img):
        g = int(preds[i])
        t = labels[i]
        print(f"  {i:3d}    {IDX_TO_LETTER[g]}      {IDX_TO_LETTER[t]}"
              f"     {'ok ' if g == t else 'MISS'}    {int(acc2[i, g]):>10,}")

    agree = (preds == np.array(labels)).mean()
    print(f"\ngolden model agrees with true labels on "
          f"{agree*100:.1f}% of these {n_img} images")
    print("(the RTL must match the GOLDEN column, not the true column)")

    # Saturation is worth knowing about: if a lot of layer-1 activations are
    # pinned at 255 the shift is too aggressive and accuracy suffers.
    sat = (a1 == 255).mean()
    print(f"\nlayer-1 activations at saturation: {sat*100:.2f}%")
    print(f"peak |acc1| over demo set: {np.abs(acc1).max():,}")
    print(f"peak |acc2| over demo set: {np.abs(acc2).max():,}")

    l1_chunks = INPUTS // N
    l2_chunks = hidden // N
    # Each layer ends by draining the pipeline for DRAIN_CYCLES+1 cycles; keep
    # this in step with the localparam of the same name in nn_accel_core.v.
    rtl = rtl_constants()
    drains = 2 * (rtl["DRAIN_CYCLES"] + 1)
    expected = expected_cycles(rtl)
    print(f"\nexpected cycles/inference = {hidden}*{l1_chunks} + "
          f"{n_class}*{l2_chunks} + {drains} drain = {expected:,}")
    # 1e8 cycles/sec divided by cycles/inference. (Dividing 1e6 by the cycle
    # count, as this line used to, silently reported 154/sec instead of 15,432.)
    print(f"  at 100 MHz that is {expected/100.0:.1f} us "
          f"({1e8/expected:,.0f} inferences/sec)")


if __name__ == "__main__":
    main()
