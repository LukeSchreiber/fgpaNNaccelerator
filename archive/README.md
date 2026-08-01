# archive/

Superseded scripts, kept because they show how the design got here. Nothing
in the shipped pipeline imports them.

| file | what it was | replaced by |
|---|---|---|
| `train_mnist_v0.py` | The original bring-up: digit MNIST, 784 → 64 → 10. Proved the training/quantization flow before the ASL dataset was involved. | `python/train_asl_v2.py` |
| `train_asl_v1.py` | First ASL model: 64 hidden units, no input normalization. | `python/train_asl_v2.py` (128 hidden, global mean/std folded into layer 1) |

Running either overwrites a checkpoint the RTL parameters do not match —
`HIDDEN` is 128 in `nn_accel_core.v`, and v1 trains 64.

The UART loopback harness (`rtl/uart_loopback_top.v`,
`constraints/uart_loopback.xdc`) is *not* archived despite being bring-up
scaffolding: both are sources of the Vivado project, and moving them would
break `nn_accel.xpr`.
