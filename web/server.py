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

import os
import sys
import threading
import time

import serial
from flask import Flask, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import predict  # noqa: E402  (needs the path set above)

PORT = os.environ.get("FPGA_PORT", predict.DEFAULT_PORT)
BAUD = predict.BAUD
TIMEOUT = 5.0

STATIC_DIR = os.path.join(ROOT, "web", "static")
MEM_DIR = os.path.join(ROOT, "mem")

app = Flask(__name__, static_folder=None)


class Board:
    """The one open serial port, and the one lock that serializes it."""

    def __init__(self, port, baud, timeout):
        self._lock = threading.Lock()
        self._ser = serial.Serial(port, baud, timeout=timeout)
        self.port = port

    def classify(self, image_bytes):
        """Send 784 pixel bytes, return (class_index, round_trip_ms).

        Raises TimeoutError if the board does not answer.
        """
        if len(image_bytes) != predict.INPUTS:
            raise ValueError(
                f"expected {predict.INPUTS} bytes, got {len(image_bytes)}")

        with self._lock:
            # Anything already buffered predates this request: a stale byte read
            # as this image's answer would be a wrong result, not an error.
            self._ser.reset_input_buffer()

            t0 = time.perf_counter()
            self._ser.write(image_bytes)
            self._ser.flush()
            reply = self._ser.read(1)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not reply:
            raise TimeoutError(
                f"no reply from {self.port} within {TIMEOUT}s -- is the board "
                f"programmed, and is this the UART rather than the JTAG port?")

        return reply[0], elapsed_ms


board = Board(PORT, BAUD, TIMEOUT)

packed = predict.read_hex_lines(os.path.join(MEM_DIR, "images_packed.mem"))
labels = predict.read_hex_lines(os.path.join(MEM_DIR, "labels.mem"))
N_IMAGES = len(packed) // predict.CHUNKS


def letter(idx):
    """Class index -> ASL letter. 24 classes: J and Z need motion."""
    if 0 <= idx < len(predict.IDX_TO_LETTER):
        return predict.IDX_TO_LETTER[idx]
    return "?"


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/info")
def info():
    """What the frontend needs to build itself: image count, port, letters."""
    return jsonify({
        "n_images": N_IMAGES,
        "port": board.port,
        "baud": BAUD,
        "letters": predict.IDX_TO_LETTER,
    })


@app.route("/api/image/<int:index>")
def image_pixels(index):
    """The 784 raw pixels, so the browser can draw what will be sent."""
    if not 0 <= index < N_IMAGES:
        return jsonify({"error": f"index out of range 0..{N_IMAGES-1}"}), 404

    return jsonify({
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
        pred, latency_ms = board.classify(image)
    except TimeoutError as e:
        return jsonify({"error": str(e)}), 504

    truth = labels[index]

    return jsonify({
        "index": index,
        "predicted_class": pred,
        "predicted_letter": letter(pred),
        "ground_truth": truth,
        "ground_truth_letter": letter(truth),
        "correct": pred == truth,
        "latency_ms": round(latency_ms, 1),
    })


if __name__ == "__main__":
    print(f"serial: {board.port} at {BAUD} baud")
    print(f"images: {N_IMAGES} from mem/images_packed.mem")
    print("open http://localhost:5000")
    # threaded=False: one request at a time, so nothing can share the port.
    app.run(host="127.0.0.1", port=5000, threaded=False, debug=False)
