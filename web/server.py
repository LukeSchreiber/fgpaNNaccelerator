"""
Web front end for the FPGA accelerator: browser -> Flask -> UART -> Basys 3.

    ./venv/bin/python web/server.py     then open http://localhost:5000

WHY THE SERIAL PORT IS A SINGLETON BEHIND A LOCK
------------------------------------------------
The board's protocol is a bare byte count: 784 bytes in, one byte back, with
nothing to say where one image stops and the next begins. Two requests
overlapping on the port would interleave their pixels, and the board would
happily classify the splice and return a byte for it. The result looks exactly
like an intermittent hardware fault, which is the most expensive kind of bug to
chase -- so the port is opened once at startup and every round trip holds a
lock for its whole duration, send and reply together.

Flask also runs with threaded=False. The lock is the real guarantee; the
single-threaded server is the belt to its braces.

The image packing is NOT reimplemented here. python/predict.py already unpacks
images_packed.mem into the byte order img_loader.v expects, and that ordering is
the one thing in this stack that fails silently -- reversed lanes still produce
a confident answer, just a wrong one. This imports that function rather than
writing a second copy to drift out of step.
"""

import base64
import io
import os
import termios
import sys
import threading
import time

import numpy as np
import serial
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import golden_check  # noqa: E402  (needs the path set above)
import predict       # noqa: E402

# None means auto-detect. FPGA_PORT pins it to a specific device.
PORT = os.environ.get("FPGA_PORT")
BAUD = predict.BAUD
TIMEOUT = 5.0

FRONTEND_DIR = os.path.join(ROOT, "frontend")
MEM_DIR = os.path.join(ROOT, "mem")

# Cycles per inference, PARSED FROM rtl/nn_accel_core.v rather than copied.
# The frontend reads this from /api/info, so a hand-maintained constant here
# would let the UI report a latency the hardware never had -- confidently
# wrong, and nothing would flag it. Change DRAIN_CYCLES or HIDDEN in the RTL
# and this follows automatically.
FPGA_CYCLES = golden_check.expected_cycles()
FPGA_CLK_HZ = 100_000_000
FPGA_MS = FPGA_CYCLES / FPGA_CLK_HZ * 1000.0

SIDE = 28                  # Sign Language MNIST is 28x28
PREVIEW_SCALE = 8          # 28 * 8 = 224 px, nearest-neighbour so pixels stay pixels
MAX_UPLOAD = 10 * 1024 * 1024

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD


class BoardUnavailable(Exception):
    """The serial link is not usable right now (unplugged, or never present)."""


# termios.error is NOT an OSError subclass, so it has to be named explicitly or
# a yanked USB cable escapes as a 500 with a stack trace.
LINK_ERRORS = (serial.SerialException, OSError, termios.error)


class Board:
    """The one serial port, and the one lock that serializes it.

    The port is opened once and held, because reopening per request would add
    tens of milliseconds and, worse, reset the FTDI line state mid-demo.

    It is NOT assumed to stay open. Unplugging the board, power-cycling it, or
    reprogramming it makes the kernel drop the device node, and every later
    write to the stale descriptor fails with EIO. So a failed round trip closes
    the handle and the next request reopens it -- plug the board back in and the
    page starts working again with no restart.
    """

    def __init__(self, port, baud, timeout):
        self._lock = threading.Lock()
        self._ser = None
        self.port = port          # None = auto-detect on every open
        self.resolved = port      # what the last open actually used
        self.baud = baud
        self.timeout = timeout

    def _open(self):
        """Caller must hold the lock. Raises BoardUnavailable.

        The port is resolved HERE rather than once at startup: unplugging the
        board and plugging it back in can hand it a different /dev/ttyUSB*
        number, and re-detecting on each open means that heals itself along
        with the reconnect.
        """
        try:
            self.resolved = predict.find_port(self.port)
        except predict.PortNotFound as e:
            self._ser = None
            raise BoardUnavailable(str(e))

        try:
            self._ser = serial.Serial(self.resolved, self.baud,
                                      timeout=self.timeout)
        except LINK_ERRORS as e:
            self._ser = None
            raise BoardUnavailable(
                f"cannot open {self.resolved}: {e}. Is the board plugged in and "
                f"powered, and is this the UART rather than the JTAG interface?")

    def _drop(self):
        """Caller must hold the lock. Discard a handle that has gone bad."""
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None

    @property
    def connected(self):
        with self._lock:
            return self._ser is not None and self._ser.is_open

    def connect(self):
        """Best-effort open at startup; never fatal."""
        with self._lock:
            try:
                self._open()
                return True
            except BoardUnavailable:
                return False

    def classify(self, image_bytes):
        """Send 784 pixel bytes, return (class_index, logit, round_trip_ms).

        Raises BoardUnavailable if the link is down, TimeoutError if the board
        is connected but silent.
        """
        if len(image_bytes) != predict.INPUTS:
            raise ValueError(
                f"expected {predict.INPUTS} bytes, got {len(image_bytes)}")

        with self._lock:
            if self._ser is None:
                self._open()          # lazy reconnect after an unplug

            try:
                # Anything already buffered predates this request: a stale byte
                # read as this image's answer would be a wrong result, not an
                # error, which is far harder to notice.
                self._ser.reset_input_buffer()

                t0 = time.perf_counter()
                self._ser.write(image_bytes)
                self._ser.flush()
                reply = self._ser.read(predict.REPLY_BYTES)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            except LINK_ERRORS as e:
                self._drop()
                raise BoardUnavailable(
                    f"lost the link to {self.resolved} mid-transfer ({e}). "
                    f"Reconnect the board; the next request will reopen it.")

        if len(reply) < predict.REPLY_BYTES:
            raise TimeoutError(
                f"no reply from {self.resolved} within {self.timeout}s -- is the "
                f"board programmed with nn_accel_top? A truncated stream is "
                f"discarded by the loader's watchdog after 50 ms, so simply "
                f"retrying is safe.")

        score = int.from_bytes(reply[1:5], "little", signed=True)
        return reply[0], score, elapsed_ms


board = Board(PORT, BAUD, TIMEOUT)

packed = predict.read_hex_lines(os.path.join(MEM_DIR, "images_packed.mem"))
labels = predict.read_hex_lines(os.path.join(MEM_DIR, "labels.mem"))
N_IMAGES = len(packed) // predict.CHUNKS


def letter(idx):
    """Class index -> ASL letter. 24 classes: J and Z need motion."""
    if 0 <= idx < len(predict.IDX_TO_LETTER):
        return predict.IDX_TO_LETTER[idx]
    return "?"


# ---------------------------------------------------------------------------
# Golden model: the same integer arithmetic the RTL implements, in NumPy.
#
# Every classification runs through it on the SAME 784 bytes the board got, so
# each result answers two separate questions instead of one:
#
#   matches_model = False  ->  the ACCELERATOR is wrong. A real hardware fault.
#   matches_model = True, but the letter is wrong for the photo
#                          ->  the accelerator is fine and the MODEL is wrong,
#                              which for an out-of-distribution photo is normal.
#
# Without this the two are indistinguishable from the browser, and "the board
# said Q" tells you nothing about whether the hardware works.
#
# forward_int is imported from golden_check.py rather than reimplemented: it is
# the reference tb_core.v is checked against, and a second copy here could
# drift and start certifying the wrong answer.
# ---------------------------------------------------------------------------
_params = np.load(os.path.join(MEM_DIR, "params.npz"))
QW1, QB1 = _params["qW1"], _params["qb1"]
QW2, QB2 = _params["qW2"], _params["qb2"]
SHIFT1 = int(_params["shift1"])


def golden(pixels):
    """Reference prediction for these exact bytes.

    Returns (class, logit, runner-up class, runner-up logit). The runner-up is
    what turns a wrong answer into a legible one: image 5 comes back as W with
    a logit 2.8% above V, which is a near-tie between the two signs that differ
    by one finger -- not the same story as a confident mistake.
    """
    px = np.frombuffer(pixels, dtype=np.uint8).astype(np.int64).reshape(1, -1)
    _, _, acc2 = golden_check.forward_int(px, QW1, QB1, SHIFT1, QW2, QB2)
    order = acc2[0].argsort()[::-1]
    a, b = int(order[0]), int(order[1])
    return a, int(acc2[0, a]), b, int(acc2[0, b])


class BadImage(Exception):
    """The upload was not a decodable image."""


def preprocess(data):
    """Uploaded file bytes -> (784 pixel bytes, the 28x28 PIL image).

    Faithful to the format the network was trained on, and nothing more:
    grayscale, square, 28x28, raw 0..255 in row-major order -- the same order
    predict.py unpacks images_packed.mem into.

    The centre crop happens BEFORE the resize because squashing a 4:3 photo
    straight into a square would distort the hand, and hand shape is the entire
    signal here. LANCZOS rather than a box filter because downsampling a
    megapixel photo to 28x28 without a decent low-pass turns fingers into
    aliasing artefacts.

    This deliberately does not normalize, threshold, or auto-contrast. The
    training set is tightly cropped, centred, evenly lit hands; an arbitrary
    photo will classify badly and the preview is there to show why. Tuning this
    to rescue bad photos would only move the mismatch somewhere harder to see.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise BadImage(f"not a readable image file ({e})")

    # Phone cameras store the sensor image and a rotation tag rather than
    # rotating the pixels. Without this a portrait photo is centre-cropped along
    # the wrong axis and the hand ends up half out of frame.
    img = ImageOps.exif_transpose(img)

    img = img.convert("L")

    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2,
                    (w - side) // 2 + side, (h - side) // 2 + side))

    img = img.resize((SIDE, SIDE), Image.LANCZOS)

    px = img.tobytes()
    if len(px) != predict.INPUTS:
        raise BadImage(f"preprocessing produced {len(px)} bytes, "
                       f"expected {predict.INPUTS}")

    return px, img


def preview_data_url(img):
    """The 28x28 upscaled with NEAREST, as a data: URL.

    Nearest-neighbour on purpose: the user should see the actual 784 pixels the
    board received, not a smoothed impression of them.
    """
    big = img.resize((SIDE * PREVIEW_SCALE, SIDE * PREVIEW_SCALE), Image.NEAREST)
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def asset(filename):
    """Serve style.css / app.js out of frontend/.

    Registered last and with a converter, so the explicit API rules above win
    the match. send_from_directory rejects paths that escape the directory.
    """
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/info")
def info():
    """What the frontend needs to build itself: image count, port, letters."""
    return jsonify({
        "n_images": N_IMAGES,
        "port": board.resolved or "auto-detect",
        "connected": board.connected,
        "fpga_cycles": FPGA_CYCLES,
        "fpga_ms": round(FPGA_MS, 4),
        "baud": BAUD,
        "letters": predict.IDX_TO_LETTER,
    })


@app.route("/api/image/<int:index>")
def image_pixels(index):
    """The 784 raw pixels, so the browser can draw what will be sent."""
    if not 0 <= index < N_IMAGES:
        return jsonify({"error": f"index out of range 0..{N_IMAGES-1}"}), 404

    return jsonify({
        "golden_class": g_idx,
        "golden_letter": letter(g_idx),
        "golden_score": g_score,
        "hardware_score": hw_score,
        "runner_up_letter": letter(g2_idx),
        "runner_up_score": g2_score,
        "margin_pct": round((g_score - g2_score) / abs(g_score) * 100, 1)
                      if g_score else 0.0,
        # Bit-exact, not just the same argmax: a drifted dot product that
        # happens to keep the same winner is exactly what the class check misses.
        "matches_model": pred == g_idx and hw_score == g_score,
        "index": index,
        "pixels": list(predict.unpack_image(packed, index)),
        "ground_truth": labels[index],
        "ground_truth_letter": letter(labels[index]),
    })


@app.route("/predict-test", methods=["POST"])
def predict_test():
    body = request.get_json(silent=True) or {}
    index = body.get("index", 0)

    if not isinstance(index, int) or not 0 <= index < N_IMAGES:
        return jsonify({"error": f"index must be an int in 0..{N_IMAGES-1}"}), 400

    image = predict.unpack_image(packed, index)

    try:
        pred, hw_score, latency_ms = board.classify(image)
    except BoardUnavailable as e:
        return jsonify({"error": str(e)}), 503
    except TimeoutError as e:
        return jsonify({"error": str(e)}), 504

    truth = labels[index]
    g_idx, g_score, g2_idx, g2_score = golden(image)

    return jsonify({
        "index": index,
        "golden_class": g_idx,
        "golden_letter": letter(g_idx),
        "golden_score": g_score,
        "hardware_score": hw_score,
        "runner_up_letter": letter(g2_idx),
        "runner_up_score": g2_score,
        "margin_pct": round((g_score - g2_score) / abs(g_score) * 100, 1)
                      if g_score else 0.0,
        # Bit-exact, not just the same argmax: a drifted dot product that
        # happens to keep the same winner is exactly what a class check misses.
        "matches_model": pred == g_idx and hw_score == g_score,
        "predicted_class": pred,
        "predicted_letter": letter(pred),
        "ground_truth": truth,
        "ground_truth_letter": letter(truth),
        "correct": pred == truth,
        "latency_ms": round(latency_ms, 1),
    })


@app.route("/predict-upload", methods=["POST"])
def predict_upload():
    upload = request.files.get("file") or request.files.get("image")
    if upload is None or upload.filename == "":
        return jsonify({"error": "no file in the request (field name: file)"}), 400

    try:
        pixels, small = preprocess(upload.read())
    except BadImage as e:
        return jsonify({"error": str(e)}), 400

    # The same locked singleton the test path uses -- one port, one lock.
    try:
        pred, hw_score, latency_ms = board.classify(pixels)
    except BoardUnavailable as e:
        return jsonify({"error": str(e)}), 503
    except TimeoutError as e:
        return jsonify({"error": str(e)}), 504

    g_idx, g_score, g2_idx, g2_score = golden(pixels)

    return jsonify({
        "filename": upload.filename,
        "golden_class": g_idx,
        "golden_letter": letter(g_idx),
        "golden_score": g_score,
        "hardware_score": hw_score,
        "runner_up_letter": letter(g2_idx),
        "runner_up_score": g2_score,
        "margin_pct": round((g_score - g2_score) / abs(g_score) * 100, 1)
                      if g_score else 0.0,
        # Bit-exact, not just the same argmax: a drifted dot product that
        # happens to keep the same winner is exactly what the class check misses.
        "matches_model": pred == g_idx and hw_score == g_score,
        "predicted_class": pred,
        "predicted_letter": letter(pred),
        "latency_ms": round(latency_ms, 1),
        "preview_png": preview_data_url(small),
    })


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": f"file exceeds {MAX_UPLOAD // (1024*1024)} MB"}), 413


if __name__ == "__main__":
    if board.connect():
        print(f"serial: {board.resolved} at {BAUD} baud (auto-detected)"
              if PORT is None else f"serial: {board.resolved} at {BAUD} baud")
    else:
        print(f"serial: no board found -- the UI will load and reconnect by "
              f"itself once one is plugged in")
    print(f"images: {N_IMAGES} from mem/images_packed.mem")
    print("open http://localhost:5000")
    # threaded=False: one request at a time, so nothing can share the port.
    app.run(host="127.0.0.1", port=5000, threaded=False, debug=False)
