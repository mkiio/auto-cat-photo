#!/usr/bin/env python3
"""
Cat Detector – single configurable capture (threaded)
----------------------------------------------------
• Raspberry Pi 4B + Pi AI Camera (Sony IMX500), Picamera2 0.3+, OpenCV, PIL
• Note: Requires displayhatmini library, which is not compatible with Raspberry Pi 5 devices.
• Live preview on Pimoroni Display HAT Mini toggled with --preview / --no-preview (default ON)
• Captures **one** photo per detection – selectable via --capture {square|crop|full}
    square : square crop (default) resized to 1024×1024
    crop   : rectangular crop around cat (with margin)
    full   : full‑frame still
"""
from __future__ import annotations
import time, sys, traceback, threading, argparse
from pathlib import Path

import numpy as np
import cv2 # Keep cv2 for other potential uses, but not for preview window
from PIL import Image, ImageDraw # Import ImageDraw for drawing on PIL images
from picamera2 import Picamera2
from picamera2.devices import IMX500

# Import Display HAT Mini library
try:
    from displayhatmini import DisplayHATMini
except ImportError:
    print("Error: displayhatmini library not found. Please install it (`pip3 install displayhatmini`).")
    sys.exit(1)


# ─────────────────────────── User settings ─────────────────────────── #
CONF_THRESHOLD   = 0.60
MIN_INTERVAL_SEC = 10      # seconds between saves
MARGIN_RATIO     = 0.20    # margin around detected box (0.20 = 10 % each side)
CROP_SIZE        = 1024    # square crop size (pixels)
SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_DIR   = SCRIPT_DIR / "photos"           # ./photos next to the script
MODEL_BLOB = (
    "/usr/share/imx500-models/"
    "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
)
COCO_CAT_ID = 17  # id for “cat” in the COCO dataset

# Display HAT Mini dimensions
DISPLAY_WIDTH = DisplayHATMini.WIDTH
DISPLAY_HEIGHT = DisplayHATMini.HEIGHT
# ────────────────────────────────────────────────────────────────────── #


def add_margin(x0, y0, x1, y1, w, h, ratio=MARGIN_RATIO):
    dx = int((x1 - x0) * ratio * 0.5)
    dy = int((y1 - y0) * ratio * 0.5)
    return (
        max(0, x0 - dx), max(0, y0 - dy),
        min(w, x1 + dx), min(h, y1 + dy),
    )


def make_square_box(x0, y0, x1, y1, w, h):
    box_w, box_h = x1 - x0, y1 - y0
    side = max(box_w, box_h)
    cx, cy = x0 + box_w // 2, y0 + box_h // 2
    half = side // 2
    nx0 = max(0, cx - half); ny0 = max(0, cy - half)
    nx1 = min(w, nx0 + side); ny1 = min(h, ny0 + side)
    # adjust if clipped
    if nx1 - nx0 < side: nx0 = max(0, nx1 - side)
    if ny1 - ny0 < side: ny0 = max(0, ny1 - side)
    return nx0, ny0, nx1, ny1


def save_photo(rgb: np.ndarray, box, out_dir: Path, mode: str):
    """Save exactly one JPEG according to *mode* (square|crop|full)."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")

        # ------ full frame ------ #
        if mode == "full":
            Image.fromarray(rgb).save(out_dir / f"cat_{ts}_full.jpg", quality=92)
            print(f"[INFO] saved cat_{ts}_full.jpg")
            return

        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return  # invalid box

        if mode == "crop":
            # rectangular crop with margin
            crop = rgb[y0:y1, x0:x1]
            Image.fromarray(crop).save(out_dir / f"cat_{ts}_crop.jpg", quality=92)
            print(f"[INFO] saved cat_{ts}_crop.jpg")
            return

        # default → square
        h_img, w_img, _ = rgb.shape
        sx0, sy0, sx1, sy1 = make_square_box(x0, y0, x1, y1, w_img, h_img)
        square = rgb[sy0:sy1, sx0:sx1]
        sq = Image.fromarray(square).resize((CROP_SIZE, CROP_SIZE), Image.Resampling.LANCZOS)
        sq.save(out_dir / f"cat_{ts}_square.jpg", quality=92)
        print(f"[INFO] saved cat_{ts}_square.jpg")

    except Exception as e:
        print(f"[WARN] failed to save photo: {e}")


class CatDetector:
    def __init__(self,
                 model_blob: str = MODEL_BLOB,
                 enable_preview: bool = True,
                 capture_mode: str = "square"):
        self.enable_preview = enable_preview
        self.capture_mode   = capture_mode  # square | crop | full

        # NN + camera -------------------------------------------------- #
        self.imx = IMX500(model_blob)
        self.cam = Picamera2(self.imx.camera_num)

        # preview (fast + inference) - Keep original preview config size for inference
        self.preview_cfg = self.cam.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        # still (full‑res)
        self.still_cfg = self.cam.create_still_configuration(
            main={"format": "RGB888"}
        )
        self.cam.configure(self.preview_cfg)
        self.cam.post_callback = self.on_request

        # state --------------------------------------------------------- #
        self.last_saved   = 0.0
        self.flash_frames = 0
        self.last_box     = (0, 0, 0, 0)
        self.pending_box  = None  # preview coords awaiting high‑res capture
        self.capture_thread = None

        # Display HAT Mini setup
        self.display_hat = None
        self.display_buffer = None
        self.display_draw = None
        if self.enable_preview:
            self.display_buffer = Image.new("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            self.display_draw = ImageDraw.Draw(self.display_buffer)
            self.display_hat = DisplayHATMini(self.display_buffer)
            self.display_hat.set_led(0.05, 0.05, 0.05) # Optional: set LED


    # ────────────────  NN callback ──────────────── #
    def on_request(self, req):
        try:
            outputs = self.imx.get_outputs(req.get_metadata())
        except Exception as e:
            print(f"[WARN] NN outputs read failed: {e}")
            return
        if not outputs or len(outputs) < 3:
            return
        boxes, scores, classes = outputs[:3]
        for i, score in enumerate(scores):
            if score < CONF_THRESHOLD or int(classes[i]) != COCO_CAT_ID:
                continue
            now = time.time()
            if now - self.last_saved < MIN_INTERVAL_SEC:
                break  # respect min‑interval
            # map to preview pixels
            rel = tuple(boxes[i])
            x, y, w, h = self.imx.convert_inference_coords(
                rel, req.get_metadata(), self.cam
            )
            x1, y1 = x + w, y + h
            # preview = req.make_array("main")  # RGB @ 640×480 - not needed in callback
            h_pre, w_pre = self.preview_cfg["main"]["size"] # Use configured preview size
            x0, y0, x1, y1 = add_margin(x, y, x1, y1, w_pre, h_pre)

            # flash + state
            self.last_saved   = now
            self.last_box     = (x0, y0, x1, y1) # Store box in preview coordinates
            self.flash_frames = 5
            # Store box in preview coordinates along with preview dimensions for scaling
            self.pending_box  = (x0, y0, x1, y1, w_pre, h_pre)


            # spin up capture thread
            if self.capture_thread is None or not self.capture_thread.is_alive():
                self.capture_thread = threading.Thread(
                    target=self._capture_highres_worker,
                    daemon=True
                )
                self.capture_thread.start()
            break

    # ────────────────  High‑res capture worker ──────────────── #
    def _capture_highres_worker(self):
        time.sleep(0.05)  # let callback fully return
        try:
            self.cam.stop()
            self.cam.configure(self.still_cfg)
            self.cam.start()

            frame = self.cam.capture_array()        # RGB888 full sensor
            full  = frame

            h_full, w_full, _ = full.shape

            # scale preview box -> full‑res box
            # Use the stored preview dimensions for scaling
            x0_pre, y0_pre, x1_pre, y1_pre, w_pre, h_pre = self.pending_box
            sx, sy = w_full / w_pre, h_full / h_pre
            box_full = (
                int(x0_pre * sx), int(y0_pre * sy),
                int(x1_pre * sx), int(y1_pre * sy),
            )
            save_photo(full, box_full, SAVE_DIR, self.capture_mode)

        except Exception as e:
            print(f"[ERROR] high‑res capture failed: {e}")
        finally:
            try:
                self.cam.stop()
                self.cam.configure(self.preview_cfg)
                self.cam.start()
            except Exception:
                pass

    # ────────────────  Main loop ──────────────── #
    def run(self):
        if self.enable_preview:
            print(f"[INFO] Display HAT Mini preview ON.")
        else:
            print("[INFO] Live preview OFF – press Ctrl‑C to quit.")

        self.cam.start()

        try:
            while True:
                # Capture preview frame as a NumPy array
                frame = self.cam.capture_array("main") # RGB preview @ 640x480

                if self.enable_preview and self.display_hat and self.display_buffer and self.display_draw:
                    # Convert NumPy array to PIL Image
                    pil_img = Image.fromarray(frame)

                    # Resize for Display HAT Mini
                    # Use LANCZOS for better quality downsampling
                    pil_img_resized = pil_img.resize((DISPLAY_WIDTH, DISPLAY_HEIGHT), Image.Resampling.LANCZOS)

                    # Draw onto the Display HAT Mini buffer
                    self.display_buffer.paste(pil_img_resized, (0, 0))

                    # Draw detection box if flashing
                    if self.flash_frames > 0:
                        # Scale the detection box coordinates from 640x480 (preview) to 320x240 (display)
                        x0_pre, y0_pre, x1_pre, y1_pre = self.last_box
                        scale_x = DISPLAY_WIDTH / self.preview_cfg["main"]["size"][0]
                        scale_y = DISPLAY_HEIGHT / self.preview_cfg["main"]["size"][1]
                        x0_disp = int(x0_pre * scale_x)
                        y0_disp = int(y0_pre * scale_y)
                        x1_disp = int(x1_pre * scale_x)
                        y1_disp = int(y1_pre * scale_y)

                        # Draw rectangle and text on the display buffer using PIL.ImageDraw
                        self.display_draw.rectangle([(x0_disp, y0_disp), (x1_disp, y1_disp)], outline=(0, 255, 0), width=2)
                        # Basic text drawing - may need font adjustments for better appearance
                        try:
                            # Attempt to load a font if available, otherwise use default
                            # Replace 'DejaVuSans.ttf' with a font file present on your system if needed
                            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # Example font path
                            font_size = 15
                            font = ImageFont.truetype(font_path, font_size)
                            self.display_draw.text((x0_disp, max(0, y0_disp - font_size - 2)), "Cat!", font=font, fill=(0, 255, 0))
                        except Exception:
                             # Fallback to default PIL font if loading fails
                             font = ImageFont.load_default()
                             self.display_draw.text((x0_disp, max(0, y0_disp - 10)), "Cat!", font=font, fill=(0, 255, 0))


                        self.flash_frames -= 1

                    # Display the buffer on the HAT
                    self.display_hat.display()

                # Add a small sleep to prevent high CPU usage when preview is off
                if not self.enable_preview:
                     time.sleep(0.01)


        except KeyboardInterrupt:
            pass
        finally:
            self.cam.stop()
            if self.enable_preview and self.display_hat:
                # Clean up Display HAT Mini resources (optional, but good practice)
                # The DisplayHATMini.__del__ method calls GPIO.cleanup() on exit.
                print("[INFO] Display HAT Mini preview OFF.")


# ──────────────────────────  CLI handling  ────────────────────────── #
def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cat detector with optional live preview on Display HAT Mini and selectable capture type."
    )

    # preview toggle -------------------------------------------------- #
    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument(
        "--preview",
        dest="preview",
        action="store_true",
        help="Enable live preview on Display HAT Mini (default)."
    )
    preview_group.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="Disable live preview."
    )
    parser.set_defaults(preview=True)

    # capture mode ---------------------------------------------------- #
    parser.add_argument(
        "--capture",
        choices=["square", "crop", "full"],
        default="square",
        help="Photo type to save on detection: square (default), crop (rectangular), or full (full‑frame)."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    try:
        CatDetector(enable_preview=args.preview,
                    capture_mode=args.capture).run()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        traceback.print_exc()
        sys.exit(1)