"""
Stage 4: send one demo image to the board over the UART and read the class back.

    ./venv/bin/python python/predict.py [index] [--port /dev/ttyUSB1]

The port is auto-detected from the USB descriptors; --port overrides.

The board runs continuously: 784 bytes in on RsRx starts an inference, one byte
comes back on RsTx with the predicted class index. No header, no framing, no
command bytes -- the byte count IS the protocol.

WHY THE IMAGE COMES FROM images_packed.mem AND NOT images.mem
-------------------------------------------------------------
images.mem is one pixel per line and would be the obvious source. Unpacking
the PACKED file instead exercises the same lane ordering the hardware uses,
in the opposite direction: pack_mem.py wrote pixel p of a chunk into bits
[8p+7:8p], so reading lane j back out of bits [8j+7:8j] must reproduce the
original raster order. If this script and img_loader.v ever disagree about
which end of the word lane 0 lives at, the image arrives mirrored in groups of
16 and the prediction is quietly wrong -- with no error anywhere. Round-tripping
through the packed file means that disagreement shows up here, on a host where
it is one print statement away, rather than on the board.

tb_img_loader.v checks the same property from the RTL side.

WHAT A MISMATCH MEANS
---------------------
preds.mem holds the integer golden model's output, not the true label. The
hardware must match preds.mem exactly -- that is the bit-accuracy claim. Where
the golden model itself is wrong (about one image in twelve, the quantized model
is ~92% accurate) labels.mem disagrees and the hardware is still correct, so
both are printed and only the preds.mem comparison sets the exit status.
"""

import argparse
import os
import re
import sys

import serial
from serial.tools import list_ports

N = 16
INPUTS = 784
CHUNKS = INPUTS // N            # 49 packed words per image

# Paths are resolved against the repo, not the working directory: this is a
# command-line tool, run from wherever the board happens to be plugged in.
MEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mem")

# The dataset drops J (9) and Z (25) -- both need motion, so neither appears as
# a static frame. Class index i is therefore the i'th letter of what remains.
RAW_LABELS = [i for i in range(26) if i not in (9, 25)]
IDX_TO_LETTER = [chr(ord("A") + raw) for raw in RAW_LABELS]

# Digilent boards carry an FTDI FT2232H with two channels on one USB device:
# interface 0 is JTAG, interface 1 is the UART this design talks to. Which
# /dev/ttyUSB* number they land on depends on what else is plugged in and in
# what order, and it changes whenever the board is re-plugged -- so the device
# is found by what it IS, not by where it happened to appear last time.
FTDI_VID = 0x0403


class PortNotFound(Exception):
    """No Digilent/FTDI serial device present."""


def _interface_index(port):
    """Which USB interface this port is, so JTAG can be told from UART.

    Linux reports location like '1-1:1.1' -- the digits after the final dot are
    the interface number. Falls back to the trailing digits of the device name,
    which preserves ttyUSB0 < ttyUSB1 ordering when location is unavailable.
    """
    m = re.search(r"\.(\d+)$", port.location or "")
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)$", port.device or "")
    return int(m.group(1)) if m else 0


def find_port(explicit=None):
    """Locate the board's UART. An explicit port always wins.

    Only one channel is usually exposed as a serial device -- on Linux the JTAG
    channel is normally claimed by Digilent's driver -- but when both appear,
    the higher interface number is the UART. Picking the first would open the
    JTAG channel, which accepts the bytes and answers nothing.
    """
    if explicit:
        return explicit

    cands = [p for p in list_ports.comports()
             if p.vid == FTDI_VID
             or "digilent" in ((p.manufacturer or "") + (p.description or "")).lower()]

    if not cands:
        raise PortNotFound(
            "no Digilent/FTDI serial device found. Is the board plugged in and "
            "powered? Pass --port to override.")

    # More than one board attached: group by USB serial number so the two
    # channels of ONE device are never confused with two devices.
    by_device = {}
    for p in cands:
        by_device.setdefault(p.serial_number or p.device, []).append(p)

    key = sorted(by_device)[0]
    chosen = sorted(by_device[key], key=_interface_index)[-1]

    if len(by_device) > 1:
        others = ", ".join(sorted(by_device)[1:])
        print(f"note: {len(by_device)} FTDI devices attached; using "
              f"{chosen.device} (serial {key}). Others: {others}. "
              f"Pass --port to choose.", file=sys.stderr)

    return chosen.device
BAUD = 921600   # matches BAUD_DIV=109 in nn_accel_top.v


def read_hex_lines(path):
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


def unpack_image(packed, index):
    """49 packed words -> 784 pixel bytes in raster order.

    Lane j of a word is bits [8j+7:8j], so lane 0 is the LOW byte and comes
    first. This is the exact inverse of pack_mem.pack_rows().
    """
    n_images = len(packed) // CHUNKS
    if not 0 <= index < n_images:
        sys.exit(f"image {index} out of range: {n_images} images in images_packed.mem")

    rows = packed[index * CHUNKS:(index + 1) * CHUNKS]

    px = []
    for word in rows:
        for j in range(N):
            px.append((word >> (8 * j)) & 0xFF)

    assert len(px) == INPUTS, f"unpacked {len(px)} bytes, expected {INPUTS}"
    return bytes(px)


def classify(port, image, timeout):
    """Send one image, return the class index the board sends back."""
    with serial.Serial(port, BAUD, timeout=timeout) as ser:
        # Anything already buffered is from an earlier run -- a stale result
        # byte would be read as this image's answer.
        ser.reset_input_buffer()

        ser.write(image)
        ser.flush()

        reply = ser.read(1)
        if not reply:
            sys.exit(
                f"no reply within {timeout}s.\n"
                f"  - is the board programmed with nn_accel_top?\n"
                f"  - is {port} the UART and not the JTAG interface?\n"
                f"  - a truncated stream is discarded after 50 ms; just retry"
            )

        # The board sends exactly one byte per image. Extra bytes mean it ran
        # more than once -- most likely BTNC was pressed, or a previous run
        # left the loader part way through an image and this transfer completed
        # it, triggering an inference on a spliced image.
        extra = ser.read_all()

    return reply[0], extra


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("index", nargs="?", type=int, default=0,
                    help="which demo image to send (default 0)")
    ap.add_argument("--port", default=None,
                    help="serial device (default: auto-detect the Digilent board)")
    # 784 bytes at 921600 8N1 is 8.5 ms and the inference is 65 us, so anything
    # past a second means the board is not answering at all.
    ap.add_argument("--timeout", type=float, default=5.0,
                    help="seconds to wait for the result byte (default 5)")
    args = ap.parse_args()

    packed = read_hex_lines(os.path.join(MEM_DIR, "images_packed.mem"))
    golden = read_hex_lines(os.path.join(MEM_DIR, "preds.mem"))
    labels = read_hex_lines(os.path.join(MEM_DIR, "labels.mem"))

    image = unpack_image(packed, args.index)

    try:
        port = find_port(args.port)
    except PortNotFound as e:
        sys.exit(str(e))

    print(f"image {args.index}: sending {len(image)} bytes to {port} "
          f"at {BAUD} baud ({len(image) * 10 / BAUD * 1000:.0f} ms)")

    pred, extra = classify(port, image, args.timeout)

    want = golden[args.index]
    truth = labels[args.index]

    def letter(i):
        return IDX_TO_LETTER[i] if 0 <= i < len(IDX_TO_LETTER) else "?"

    print("")
    print(f"  hardware      {pred:2d}  {letter(pred)}")
    print(f"  golden model  {want:2d}  {letter(want)}")
    print(f"  true label    {truth:2d}  {letter(truth)}")
    print("")

    if extra:
        print(f"warning: {len(extra)} unexpected extra byte(s) after the result: "
              f"{extra.hex()}")

    if pred != want:
        print("MISMATCH -- hardware disagrees with the golden model")
        sys.exit(1)

    print("match: hardware agrees with the golden model", end="")
    print("" if want == truth else "  (both differ from the true label)")


if __name__ == "__main__":
    main()
