# System Overview

How the accelerator works, why it is built this way, and what was measured.
For build and run instructions see the [README](../README.md).

---

## 1. What it does

A Basys 3 (Xilinx Artix-7 XC7A35T) classifies American Sign Language letters
from 28×28 grayscale images streamed over UART. A 784 → 128 → 24 MLP runs
entirely in fixed-point on the FPGA: int8 weights, int32 accumulators, no
floating point anywhere in the hardware.

```
                    ┌──────────────────────── Basys 3 / XC7A35T ───────────────────────┐
                    │                                                                  │
  host ──784 B──────┼──> uart_rx ──> img_loader ──┐                                    │
       921600 8N1   │   (8N1 rx,     (packs 16 B   │                                    │
                    │    framing      per word,    │                                    │
                    │    check)       50 ms        │                                    │
                    │                 watchdog)    │                                    │
                    │                              v                                    │
                    │                        nn_accel_core                              │
                    │        ┌───────────────────────────────────────────┐              │
                    │        │  rom_sync ×4 (w1, w2, b1, b2)             │              │
                    │        │       │                                    │              │
                    │        │       v                                    │              │
                    │        │  mac_array (16 × mac_unit, 16 DSP48E1)     │              │
                    │        │       │  split reduction: 8+8 → registered  │              │
                    │        │       v                                    │              │
                    │        │  requantize (>>> SHIFT1, ReLU, saturate)   │              │
                    │        │       │                                    │              │
                    │        │       v                                    │              │
                    │        │  act_mem (128 B) ──> layer 2 ──> argmax     │              │
                    │        └───────────────────────────────────────────┘              │
                    │                              │                                    │
  host <──5 B───────┼──── uart_tx <────────────────┤                                    │
   class + logit    │                              └──> seven_seg (letter + class)      │
                    │                                   LD15 done · LD14 busy           │
                    │                                   LD13 framing error (sticky)     │
                    └──────────────────────────────────────────────────────────────────┘
```

---

## 2. Measured results

| | |
|---|---|
| Accuracy (INT8, 7,172 test images) | **92.40%** |
| Latency | **6,480 cycles = 64.8 µs** @ 100 MHz |
| Throughput, compute-bound | 15,432 inferences/sec |
| Throughput, over the link | 117 images/sec (8.5 ms per 784-byte transfer) |
| Timing | **WNS +0.688 ns**, 0 failing endpoints of 6,363 |
| DSP48E1 | 16 / 90 (17.8%) — one per MAC lane |
| Block RAM | 37 / 50 (74%) |
| LUTs / FFs | 1,355 (6.5%) / 1,903 (4.6%) |

The link, not the accelerator, sets the frame rate: the transfer is 130× the
inference. This is why no attempt is made to overlap load and compute — the
win would be invisible.

---

## 3. The network and its quantization

Trained in PyTorch on [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
(`python/train_asl_v2.py`), then quantized by `python/quantize.py`.

**24 classes, not 26.** J and Z require motion and have no static frame.

**Normalization is folded into layer 1.** Training fed `fc1` the value
`(p/255 − MEAN)/STD`. That is affine, so it collapses into the weights:

```
W·((p/255 − μ)/σ) + b  =  (W/(255σ))·p  +  (b − (μ/σ)·rowsum(W))
```

The hardware therefore consumes the **raw pixel byte** and never sees a
floating-point number. The constants are derived from the training set at
export time, not transcribed by hand — a stale copy would silently mis-scale
every weight, and nothing downstream would notice.

**Requantization between layers is an arithmetic shift**, with the shift
amount chosen from the 99.9th percentile of observed layer-1 accumulator
magnitudes. `requantize.v` clips to `[0, 255]`, which fuses ReLU and
saturation into one step.

---

## 4. Datapath

**16 MAC lanes.** Lane *i* accumulates the terms at positions *i*, *i+16*,
*i+32*… of one dot product. Weights are packed 16-per-word into 128-bit BRAM
rows, so one read feeds every lane, and consecutive addresses walk a single
neuron to completion — the address generator just increments, with no strided
access and no multiplier in the address path.

The alternative (one lane per output neuron, broadcasting the activation)
avoids the reduction entirely but needs a strided weight fetch and produces
16 results at once, which complicates requantization downstream.

**DSP inference is structural, not a hint.** `mac_unit.v` registers the
product before the accumulate, matching the DSP48E1's MREG template. Writing
the obvious `acc <= (clr ? init : acc) + a*w;` puts multiply and add in one
combinational hop, which is legal, slow, and outside the pattern Vivado's
inference engine matches — measured at **1,826 LUTs and 0 DSPs**. The
`use_dsp` attribute alone does not fix it; the mismatch is structural.

### Pipeline alignment

The whole design hinges on this:

```
cycle k    address issued to ROM
cycle k+1  ROM data valid on the MAC inputs      (rom_sync latency 1)
cycle k+2  product registered                    (mac_unit stage 1)
cycle k+3  accumulator updated                   (mac_unit stage 2)
cycle k+4  half reductions registered            (mac_array psum_lo/psum_hi)
cycle k+5  reduction result registered           (sum_r)
```

`en`/`clr` are delayed **one** cycle from address generation so they arrive
with the data they belong to; the capture strobe is delayed **five**. The
neuron index rides the same delay chain, so a captured result knows which
output it belongs to.

**Neurons stream back to back.** Addresses run continuously and `clr`
restarts the accumulator every CHUNKS cycles, so neuron *i+1* begins
accumulating before neuron *i*'s sum has been read. Neuron *i*'s sum is valid
for exactly one cycle, and the capture register samples on the edge ending
that cycle — reading the old accumulator value while the new one is written.
Zero gap cycles. A per-neuron drain would have been easier to reason about and
cost 128 × 3 = 384 extra cycles in layer 1, about 6%.

### Why the reduction is split

Summing 16 lanes in one cycle was a 16-input 32-bit adder: **17 logic levels**
from the DSP P register to `sum_r`, and the longest path in the design. It now
reduces lanes 0–7 and 8–15 into two *registered* partial sums, and the single
add that combines them feeds `sum_r` on the following cycle.

Cost: one extra stage on the capture chain and one drain cycle per layer —
**2 cycles out of 6,480**, or 0.03% of runtime, to halve the longest path.

### Latency budget

```
layer 1   128 × 49 = 6,272 cycles
layer 2    24 ×  8 =   192 cycles
drains      2 ×  8 =    16 cycles   (DRAIN_CYCLES+1 each)
                     ─────────────
                     6,480 cycles = 64.8 µs @ 100 MHz
```

---

## 5. The link

**Host → board:** 784 raw pixel bytes. No header, no length field, no command
bytes — the byte count *is* the protocol. Arrival of the 784th byte is itself
the request to classify.

**Board → host:** 5 bytes — the class index, then the winning logit as an
int32 little-endian.

The logit is not decoration. A matching class only says the argmax fell the
same way; the logit is bit-exact or it is not. That is the standard
`tb_core.v` holds the RTL to in simulation, and shipping it makes the same
check available on the live path.

**Byte packing must match `pack_mem.py` exactly.** The MAC array slices its
input as `a_flat[j*8 +: 8]`, so lane *j* is byte *j* counting from the LOW
end. `img_loader` shifts each arriving byte in at the *top* of a 128-bit
register and right-shifts, which walks the first byte of a group down to bits
[7:0] exactly as the group closes. Get this backwards and every group of 16
pixels is mirrored: accuracy collapses to roughly chance, and **nothing
anywhere reports an error**.

**Backpressure:** while `img_ready` is high the buffer holds an image the core
is reading, so incoming bytes are dropped rather than overwriting it. Dropping
is the safe failure — a torn image would be scored and reported as real.

**Idle watchdog:** a stream that stops part way is discarded after 50 ms of
silence. Without it the loader waits forever for bytes that never come and the
*next* image silently completes the abandoned one, leaving the buffer holding
the tail of one picture and the head of another. Recovery used to require
pressing BTNU — a hardware reset for a software protocol error.

**Framing errors** are flagged (`LD13`, sticky) but the byte is still
delivered. On a 784-byte image one bad byte is one bad pixel and almost
certainly the same class, whereas dropping it leaves the image one short and
costs a 50 ms watchdog plus a host timeout. Corruption should be *visible*;
degrading a working link to achieve that is not a good trade.

---

## 6. Verification

The claim is **bit-exactness against an integer golden model**, not accuracy.
The quantized model is ~92% accurate, so checking against true labels would
fail on correct hardware.

| testbench | covers |
|---|---|
| `tb_core.v` | full inference vs. golden model — class, **logit**, and cycle count, 16 images |
| `tb_top_uart.v` | the whole chain over the wire: real 8N1 frames in, 5-byte reply decoded |
| `tb_img_loader.v` | byte packing vs. `images_packed.mem`, backpressure, watchdog |
| `tb_mac_array.v` | dot products, bias fusion, negative weights |
| `tb_mac_unit.v` | pipeline latency, signedness |
| `tb_datapath.v` | requantize saturation, argmax ties |
| `tb_uart_rx.v` / `tb_uart_tx.v` | 8N1 framing at the deployed 921600 baud |

`make test` runs all eight.

**Why the logit and not just the class.** Deleting one stage from the capture
chain — so every neuron is read one cycle early, before its last 16 terms land
— changes the predicted class on **exactly one of sixteen images and leaves
the other fifteen passing**. A class-only check calls that a pass. The logit
catches it on image 0. This is the failure mode that matters in a pipelined
design: not a crash, a quiet 4%-accuracy build.

**Why an end-to-end testbench.** `BAUD_DIV` and `BAUD_HALF` are localparams in
`nn_accel_top`, so no module-level test can see them — `uart_rx` passes its own
tests at any divisor you hand it. `tb_top_uart.v` derives its stimulus rate
from the same constant the hardware uses. It found an X-propagation bug on its
first run: the `btnC` debounce registers had no reset, so `btnC_stable` was X,
`if (X)` took the else branch and never assigned it, and the X reached
`start_core` and stuck there (`x && x = x`). Xilinx flops configure to 0 so the
board was unaffected — but every top-level simulation was dead on arrival.

**Host-side round trip.** `predict.py` deliberately unpacks
`images_packed.mem` rather than reading `images.mem`, exercising the same lane
ordering the hardware uses in the opposite direction. A disagreement surfaces
on the host, where it is one print statement away, instead of as a silent
4%-accuracy build.

**Live self-check.** The web server runs the same integer model on the same
784 bytes it sent, so every classification reports `matches_model` separately
from `correct`. That separates *the accelerator is wrong* (a real fault) from
*the model is wrong about this picture* (normal, and expected on photos).

---

## 7. Worked example: image 5

The demo set's one disagreement, and a good illustration of the above.

```
1. class 21 W  logit 8,949   <- hardware AND golden model
2. class 20 V  logit 8,694   <- true label
3. class  9 K  logit 4,471

margin: 255 of 8,949 = 2.8%
```

The hardware is bit-exact (`matches_model: true`); the *model* is wrong
(`correct: false`). V and W differ by one finger — index+middle versus
index+middle+ring — so at 28×28 grayscale this is a near-tie, while third
place is twice as far away. The network is confidently choosing between the
two signs a human would also confuse.

On the seven-segment display this reads `-  21`: W is one of five letters
(K, M, V, W, X) that need diagonals and cannot be drawn honestly with seven
segments, so they show a dash. The two digits are the authoritative output.
The tempting hacks are worse — `u` for V and `n` for M collide with U (19) and
N (12), turning "cannot draw this" into a confident wrong letter.

---

## 8. Known limitations

- **No checksum on the link.** A dropped byte is caught by the watchdog (the
  image never completes); a *corrupted* byte is flagged by the framing check
  only if it broke the frame. A CRC would close the rest.
- **No load/compute overlap.** The loader drops bytes while `img_ready` is
  high. Double-buffering would recover 64.8 µs out of 8.5 ms — 0.8%.
- **Layer 2 underuses the array.** 8 chunks per neuron versus 49, so the
  16 lanes idle 84% of the time during layer 2 — 3% of total cycles.
- **`N_IMAGES`/`img_sel` are vestigial.** The core still supports a 16-image
  address space; the top ties `img_sel` to 0 since the loader holds exactly one.
- **The capture chain is hand-maintained.** Adding a pipeline stage anywhere
  between the MAC inputs and `sum_r` requires adding one to the `c1..c4` chain
  and one to `DRAIN_CYCLES`. Getting it wrong does not fail timing — it files
  results under the wrong neuron. The logit check in `tb_core.v` is the guard.

---

## 9. Repository map

```
rtl/        Verilog-2001 source; nn_accel_top.v is the Basys 3 top level
tb/         self-checking testbenches (make test)
python/     training, quantization, packing, golden model, host CLI
web/        Flask server: serial singleton behind a lock, live golden check
frontend/   browser UI, no framework
mem/        packed weights, demo images, golden predictions and logits
bit/        implemented bitstream (+ .mcs/.prm for SPI flash boot)
docs/       this file, hardware verification log, sample image
archive/    superseded training scripts, with a README saying what replaced them
```
