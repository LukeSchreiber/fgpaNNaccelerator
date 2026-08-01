"""
Stage 2b: pack the quantized parameters into N-wide memory images.

Run after quantize.py. Reads mem/params.npz, writes hex files that
rom_sync.v loads with $readmemh.

WHY PACKING IS NEEDED
---------------------
The MAC array consumes N weights per cycle. A single BRAM port is at most
36 bits wide, so N=16 eight-bit weights (128 bits) cannot come from one
port at one address. Instead each memory word holds all N weights for one
cycle, and Vivado spreads the array across as many physical BRAMs as the
width demands.

BIT ORDER -- get this wrong and everything silently breaks
----------------------------------------------------------
mac_array slices its input as w_flat[j*8 +: 8], i.e. lane j occupies bits
[8j+7 : 8j]. Lane 0 is therefore the LOW byte, which is the LAST pair of
hex characters on the line. The packer builds an integer with lane j
shifted left by 8j and formats that, so the ordering follows from the
arithmetic rather than from string concatenation (which is where this
usually goes wrong).

ADDRESS LAYOUT
--------------
layer 1:  addr = neuron * (784/N) + chunk      -> 128 * 49 = 6272 rows
layer 2:  addr = neuron * (HIDDEN/N) + chunk   ->  24 *  8 =  192 rows
images:   addr = image  * (784/N) + chunk      ->  16 * 49 =  784 rows

Consecutive addresses walk one neuron's dot product to completion, so the
controller just increments -- no strided addressing, no multiplier in the
address path.
"""

import os
import numpy as np

N = 16                      # MAC lanes; must divide 784 and HIDDEN
INPUTS = 784
MEM_DIR = "./mem"

# 784 = 2^4 * 7^2, so N must be one of 1,2,4,7,8,14,16,28,49,...
# N=32 does NOT divide 784; that sweep point needs zero-padding to 800.
assert INPUTS % N == 0, f"N={N} does not divide {INPUTS}; pad the input vector"


def pack_rows(mat, n_lanes, width_bits=8):
    """mat: (neurons, inputs) int array -> list of packed ints, row-major.

    Row (i, c) holds mat[i, c*n_lanes : (c+1)*n_lanes], lane j in bits
    [8j+7 : 8j].
    """
    neurons, inputs = mat.shape
    assert inputs % n_lanes == 0
    chunks = inputs // n_lanes
    mask = (1 << width_bits) - 1

    rows = []
    for i in range(neurons):
        for c in range(chunks):
            word = 0
            for j in range(n_lanes):
                word |= (int(mat[i, c * n_lanes + j]) & mask) << (width_bits * j)
            rows.append(word)
    return rows


def write_hex(path, rows, width_bits):
    digits = (width_bits + 3) // 4
    with open(path, "w") as f:
        for v in rows:
            f.write(f"{v:0{digits}x}\n")
    return len(rows)


def main():
    p = np.load(f"{MEM_DIR}/params.npz")
    qW1, qb1 = p["qW1"], p["qb1"]
    qW2, qb2 = p["qW2"], p["qb2"]
    shift1 = int(p["shift1"])

    hidden, inputs = qW1.shape
    n_class, hidden2 = qW2.shape
    assert hidden == hidden2
    assert inputs == INPUTS
    assert hidden % N == 0, f"N={N} must divide HIDDEN={hidden}"

    word_bits = 8 * N

    # ---- layer 1 weights ------------------------------------------------
    rows = pack_rows(qW1, N)
    n1 = write_hex(f"{MEM_DIR}/w1_packed.mem", rows, word_bits)

    # ---- layer 2 weights ------------------------------------------------
    rows = pack_rows(qW2, N)
    n2 = write_hex(f"{MEM_DIR}/w2_packed.mem", rows, word_bits)

    # ---- biases: one per neuron, 32-bit, no packing needed --------------
    nb1 = write_hex(f"{MEM_DIR}/b1_packed.mem",
                    [int(v) & 0xFFFFFFFF for v in qb1], 32)
    nb2 = write_hex(f"{MEM_DIR}/b2_packed.mem",
                    [int(v) & 0xFFFFFFFF for v in qb2], 32)

    # ---- demo images ----------------------------------------------------
    # quantize.py wrote one pixel per line; re-read and pack N-wide.
    with open(f"{MEM_DIR}/images.mem") as f:
        px = [int(line.strip(), 16) for line in f if line.strip()]
    n_img = len(px) // INPUTS
    img = np.array(px, dtype=np.int32).reshape(n_img, INPUTS)
    rows = pack_rows(img, N)
    ni = write_hex(f"{MEM_DIR}/images_packed.mem", rows, word_bits)

    with open(f"{MEM_DIR}/labels.mem") as f:
        n_lab = sum(1 for line in f if line.strip())

    # ---- report ---------------------------------------------------------
    bits = (n1 + n2 + ni) * word_bits + (nb1 + nb2) * 32
    print(f"N = {N}   word = {word_bits} bits\n")
    print(f"  w1_packed.mem      {n1:6,} rows x {word_bits:3} b")
    print(f"  w2_packed.mem      {n2:6,} rows x {word_bits:3} b")
    print(f"  images_packed.mem  {ni:6,} rows x {word_bits:3} b  ({n_img} images)")
    print(f"  b1_packed.mem      {nb1:6,} rows x  32 b")
    print(f"  b2_packed.mem      {nb2:6,} rows x  32 b")
    print(f"  labels.mem         {n_lab:6,} rows")
    print(f"\ntotal {bits:,} bits = {100.0*bits/1_843_200:.1f}% of "
          f"the 35T's 1,800 Kbit BRAM")

    print(f"\nRTL parameters:")
    print(f"  N          = {N}")
    print(f"  HIDDEN     = {hidden}")
    print(f"  NUM_CLASS  = {n_class}")
    print(f"  SHIFT1     = {shift1}")
    print(f"  L1_CHUNKS  = {INPUTS // N}       // cycles per layer-1 neuron")
    print(f"  L2_CHUNKS  = {hidden // N}       // cycles per layer-2 neuron")
    print(f"  W1_DEPTH   = {n1}")
    print(f"  W2_DEPTH   = {n2}")
    print(f"  IMG_DEPTH  = {ni}")


if __name__ == "__main__":
    main()
