# fpga-nn-accel

An INT8 neural network accelerator on a Basys 3 (Xilinx Artix-7 XC7A35T) that
classifies American Sign Language letters from images streamed in over UART.

Send 784 pixel bytes to the board; it returns the predicted class in 64.8 µs
and shows the letter on the seven-segment display.

```
host ──784 bytes──> uart_rx ─> img_loader ─> nn_accel_core ─> argmax ─> uart_tx ──1 byte──> host
                                                  │
                                                  └─> seven_seg
```

## Results

| | |
|---|---|
| Accuracy (INT8, 7,172 test images) | **92.40%** |
| Latency | **6,480 cycles = 64.8 µs** @ 100 MHz |
| Throughput | 15,432 inferences/sec (compute-bound), 14/sec over the UART |
| Timing | **WNS +0.262 ns**, 0 failing endpoints of 6,162 |
| DSPs | 16 / 90 (17.8%) — one per MAC lane |
| Block RAM | 37 / 50 (74%) |
| LUTs / FFs | 1,315 (6.3%) / 1,835 (4.4%) |

The link, not the accelerator, sets the frame rate: 784 bytes at 115200 baud is
68 ms, about a thousand times the inference itself.

## The network

A 784 → 128 → 24 MLP trained in PyTorch on
[Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist),
then quantized to int8 weights with int32 accumulators. 24 classes, not 26 — J
and Z require motion and have no static frame.

Quantization folds the `/255` rescale and input normalization into layer 1, so
the hardware consumes raw pixel bytes and never sees a floating-point number.
Requantization between layers is an arithmetic shift, chosen at export time
from the 99.9th percentile of observed accumulator magnitudes.

## Architecture

**16 MAC lanes.** Lane *i* accumulates terms *i*, *i+16*, *i+32*… of one dot
product. Weights are packed 16-per-word into 128-bit BRAM rows so a single
read feeds every lane, and consecutive addresses walk one neuron to completion
— the address generator just increments.

**Everything is pipelined, which is the whole design problem.** ROM read, the
multiply, the accumulate, two reduction stages: five cycles from issuing an
address to a finished dot product. Control signals ride matching delay chains
so each result is filed under the neuron it belongs to. Neurons stream
back-to-back with no gap — neuron *i+1* starts accumulating before neuron *i*'s
sum has been read.

**The adder reduction is split.** Summing 16 lanes in one cycle was 17 logic
levels and the longest path in the design. It now reduces lanes 0–7 and 8–15
into two registered partial sums, then adds those. Cost: one extra drain cycle
per layer, 2 cycles out of 6,480.

## Verification

The RTL is checked bit-exactly against an integer golden model in NumPy —
class index, **winning logit, and cycle count** for all 16 demo images.

Checking the class alone is not enough. Deleting one stage from the capture
chain, so every neuron is read one cycle early before its last 16 terms land,
changes the predicted class on exactly **one of 16 images and leaves the other
15 passing**. The logit check catches it on image 0. This is the failure mode
that matters in a pipelined design: not a crash, a quiet 4%-accuracy build.

```bash
iverilog -o tb_core.vvp tb/tb_core.v rtl/nn_accel_core.v rtl/rom_sync.v \
    rtl/mac_array.v rtl/mac_unit.v rtl/requantize.v rtl/argmax.v && vvp tb_core.vvp
```

| testbench | covers |
|---|---|
| `tb_core.v` | full inference vs. golden model, 16 images |
| `tb_img_loader.v` | UART byte packing vs. `images_packed.mem`, backpressure |
| `tb_mac_array.v` | dot products, bias fusion, negative weights |
| `tb_mac_unit.v` | pipeline latency, signedness |
| `tb_datapath.v` | requantize saturation, argmax ties |
| `tb_uart_rx.v` / `tb_uart_tx.v` | 8N1 framing at 115200 baud |

## Running it

Program `bit/nn_accel_top.bit` onto a Basys 3, then:

```bash
python python/predict.py 0 --port /dev/ttyUSB1
```

```
image 0: sending 784 bytes to /dev/ttyUSB1 at 115200 baud (68 ms)

  hardware       6  G
  golden model   6  G
  true label     6  G

match: hardware agrees with the golden model
```

`BTNC` re-runs the buffered image without a resend; `BTNU` resets.

To rebuild the weights from scratch (needs the dataset CSVs, not in this repo):

```bash
python python/train_asl.py     # PyTorch training
python python/quantize.py      # fold normalization, quantize to int8
python python/pack_mem.py      # pack 16-wide for the BRAMs
python python/golden_check.py  # export preds.mem + scores.mem for the testbench
```

## Layout

```
rtl/     Verilog-2001 source; nn_accel_top.v is the Basys 3 top level
tb/      self-checking testbenches
python/  training, quantization, packing, golden model, host script
mem/     packed weights and demo images ($readmemh at configuration)
bit/     implemented bitstream
```

Pin constraints are in `rtl/basys3.xdc`. UART is 115200 8N1 on RsRx (B18) /
RsTx (A18).
