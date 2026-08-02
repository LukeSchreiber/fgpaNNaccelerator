# fpga-nn-accel

An INT8 neural network accelerator on a Basys 3 (Xilinx Artix-7 XC7A35T) that
classifies American Sign Language letters from images streamed in over UART.

Send 784 pixel bytes to the board; it returns the predicted class and its
winning logit in 64.8 µs, and shows the letter on the seven-segment display.

```
host ──784 bytes──> uart_rx ─> img_loader ─> nn_accel_core ─> argmax ─> uart_tx ─5 bytes─> host
                                                  │
                                                  └─> seven_seg
```

## Results

| | |
|---|---|
| Accuracy (INT8, 7,172 test images) | **92.40%** |
| Latency | **6,480 cycles = 64.8 µs** @ 100 MHz |
| Throughput | 15,432 inferences/sec (compute-bound), 117/sec over the UART |
| Timing | **WNS +0.262 ns**, 0 failing endpoints of 6,162 |
| DSPs | 16 / 90 (17.8%) — one per MAC lane |
| Block RAM | 37 / 50 (74%) |
| LUTs / FFs | 1,315 (6.3%) / 1,835 (4.4%) |

The link, not the accelerator, sets the frame rate: 784 bytes at 921600 baud is
8.5 ms, still 130x the inference itself.

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

`tb_top_uart.v` is the only test that exercises the baud divisors, which are
localparams in `nn_accel_top` and therefore invisible to every module-level
test — `uart_rx` passes its own tests at any divisor you hand it. It caught an
X-propagation bug that made the core unstartable in simulation.

Checking the class alone is not enough. Deleting one stage from the capture
chain, so every neuron is read one cycle early before its last 16 terms land,
changes the predicted class on exactly **one of 16 images and leaves the other
15 passing**. The logit check catches it on image 0. This is the failure mode
that matters in a pipelined design: not a crash, a quiet 4%-accuracy build.

```bash
make test          # all 8 testbenches, pass/fail summary
make tb_core       # just one
make lint          # elaborate the Basys 3 top level
```

Run from the repo root: the testbenches load `mem/*.mem` relative to the
working directory. Building elsewhere? `make test MEM_DIR=/abs/path/to/mem`.

| testbench | covers |
|---|---|
| `tb_core.v` | full inference vs. golden model, 16 images |
| `tb_img_loader.v` | UART byte packing vs. `images_packed.mem`, backpressure |
| `tb_mac_array.v` | dot products, bias fusion, negative weights |
| `tb_mac_unit.v` | pipeline latency, signedness |
| `tb_datapath.v` | requantize saturation, argmax ties |
| `tb_uart_rx.v` / `tb_uart_tx.v` | 8N1 framing at the deployed 921600 baud |
| `tb_top_uart.v` | **the whole chain over the wire**: real UART frames into `RsRx`, decoded reply off `RsTx` |

## Running it

Program `bit/nn_accel_top.bit` onto a Basys 3, then:

```bash
python python/predict.py 0        # port is auto-detected
```

```
image 0: sending 784 bytes to /dev/ttyUSB2 at 921600 baud (9 ms)

  hardware       6  G   logit   11,656
  golden model   6  G   logit   11,656
  true label     6  G

match: hardware agrees with the golden model
```

The reply is 5 bytes: the class, then the winning logit as an int32. The logit
is what makes the check mean something — a matching class only says the argmax
fell the same way, while the logit is bit-exact or it is not. That is the
standard `tb_core.v` holds the RTL to in simulation, now available live.

`BTNC` re-runs the buffered image without a resend; `BTNU` resets.
`LD13` is a sticky UART framing-error flag — lit means the link mistimed a
frame since reset. The byte is still delivered (one bad pixel in 784 beats
a stalled transfer), so the LED is the only place that corruption shows.

To rebuild the weights from scratch (needs the dataset CSVs, not in this repo):

```bash
python python/train_asl_v2.py  # PyTorch training (v2: 128 hidden, normalized)
python python/quantize.py      # fold normalization, quantize to int8
python python/pack_mem.py      # pack 16-wide for the BRAMs
python python/golden_check.py  # export preds.mem + scores.mem for the testbench
```

## Building the bitstream

The RTL loads its weights with `$readmemh` using **bare filenames**, so the
`.mem` files must be design sources of the Vivado project — there are no
absolute paths in the repo. In the Tcl console with `nn_accel.xpr` open:

```tcl
set repo [file normalize [get_property DIRECTORY [current_project]]/..]
add_files -norecurse -fileset sources_1 [list \
  $repo/mem/w1_packed.mem $repo/mem/w2_packed.mem \
  $repo/mem/b1_packed.mem $repo/mem/b2_packed.mem]
update_compile_order -fileset sources_1

reset_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
```

Add them **referenced in place, not copied** — a copy inside the project
directory becomes a second set of weights that silently goes stale the next
time `pack_mem.py` runs.

This step is not optional and it fails quietly: an unresolvable `$readmemh`
filename leaves the BRAMs zero-initialized, and the board then runs happily and
classifies everything as one class with no error anywhere. After programming,
confirm with `python python/predict.py 0` — a zero-weight build cannot match
the golden model.

To simulate the top level directly, override the paths instead:

```verilog
nn_accel_top #(.W1_FILE("mem/w1_packed.mem"), .W2_FILE("mem/w2_packed.mem"),
               .B1_FILE("mem/b1_packed.mem"), .B2_FILE("mem/b2_packed.mem"))
    dut (...);
```

## Layout

```
rtl/     Verilog-2001 source; nn_accel_top.v is the Basys 3 top level
tb/      self-checking testbenches
python/  training, quantization, packing, golden model, host script
mem/     packed weights and demo images ($readmemh at configuration)
bit/     implemented bitstream
```

Pin constraints are in `rtl/basys3.xdc`. UART is 921600 8N1 on RsRx (B18) /
RsTx (A18).

The serial device is found from its USB descriptors (FTDI VID 0x0403 /
Digilent), not by a hardcoded path — the board lands on a different
`/dev/ttyUSB*` depending on what else is plugged in. On an FT2232H the
higher interface number is the UART; interface 0 is JTAG. Override with
`--port`, or `FPGA_PORT` for the web server.
