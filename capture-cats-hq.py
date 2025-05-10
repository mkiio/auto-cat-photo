#!/usr/bin/env python3
"""
Cat Detector – AI Preview & HQ Capture
---------------------------------------
• Raspberry Pi 5 + Pi AI Camera (Sony IMX500 as cam0 for detection/preview)
• Raspberry Pi HQ Camera (Sony IMX477 as cam1 for capture)
• Picamera2 0.3+, OpenCV, PIL
• Live-preview window (from cam0) toggled with --preview / --no-preview (default ON).
• When a cat is detected on cam0, a full-frame photo is captured by cam1.
  - Saving this photo is toggled with --save-photo / --no-save-photo (default ON).
  - Saved as cat_{timestamp}_HQ.jpg
"""
from __future__ import annotations
import time, sys, traceback, threading, argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from picamera2 import Picamera2
from picamera2.devices import IMX500

# ─────────────────────────── User settings ─────────────────────────── #
CONF_THRESHOLD   = 0.50
MIN_INTERVAL_SEC = 10      # seconds between saves
MARGIN_RATIO     = 0.20    # margin around detected box for preview (0.20 = 10 % each side)
SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_DIR   = SCRIPT_DIR / "photos"           # ./photos next to the script
MODEL_BLOB = (
    "/usr/share/imx500-models/"
    "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
)
COCO_CAT_ID = 17  # id for “cat” in the COCO dataset
# ────────────────────────────────────────────────────────────────────── #


def add_margin(x0, y0, x1, y1, w, h, ratio=MARGIN_RATIO):
    dx = int((x1 - x0) * ratio * 0.5)
    dy = int((y1 - y0) * ratio * 0.5)
    return (
        max(0, x0 - dx), max(0, y0 - dy),
        min(w, x1 + dx), min(h, y1 + dy),
    )

def save_hq_photo(rgb_hq: np.ndarray, out_dir: Path, timestamp: str):
    """Save a full-frame HQ JPEG using provided timestamp."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / f"cat_{timestamp}_HQ.jpg"
        Image.fromarray(rgb_hq).save(filepath, quality=95) # Higher quality for HQ
        print(f"[INFO] saved {filepath.name}")
    except Exception as e:
        print(f"[WARN] failed to save HQ photo: {e}")
        traceback.print_exc()


class CatDetector:
    def __init__(self,
                 model_blob: str = MODEL_BLOB,
                 enable_preview: bool = True,
                 enable_photo_save: bool = True): # New parameter name
        self.enable_preview = enable_preview
        self.enable_photo_save = enable_photo_save # Controls saving of HQ photo
        
        self.cam = None # AI Camera (cam0)
        self.cam_hq = None # HQ Camera (cam1)
        self.still_cfg_hq = None

        # NN + camera (cam0 - AI Camera for Preview/Detection) -------------- #
        print("[INFO] Initializing AI Camera (cam0)...")
        self.imx = IMX500(model_blob)
        self.cam = Picamera2(self.imx.camera_num) # Should be cam0

        self.preview_cfg = self.cam.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        # No separate still_cfg for cam0 needed as it doesn't save photos
        self.cam.configure(self.preview_cfg)
        self.cam.post_callback = self.on_request
        print("[INFO] AI Camera (cam0) initialized for preview and detection.")

        # HQ Camera (cam1 - Sony IMX477 for Capture) ------------------------ #
        if self.enable_photo_save: # Only init HQ camera if we plan to save photos
            try:
                print("[INFO] Initializing HQ Camera (cam1)...")
                self.cam_hq = Picamera2(1) # Explicitly camera 1
                self.still_cfg_hq = self.cam_hq.create_still_configuration(
                     main={"format": "RGB888"} # Ensure RGB for PIL
                )
                self.cam_hq.configure(self.still_cfg_hq)
                print("[INFO] HQ Camera (cam1) initialized.")
            except Exception as e:
                print(f"[WARN] Failed to initialize HQ Camera (cam1): {e}. Disabling photo capture for this session.")
                self.cam_hq = None
                self.enable_photo_save = False # Auto-disable if init fails
        else:
            print("[INFO] Photo saving (HQ capture) disabled by configuration.")

        # state --------------------------------------------------------- #
        self.last_saved   = 0.0
        self.flash_frames = 0
        self.last_box     = (0, 0, 0, 0) # For preview overlay
        self.capture_thread = None

    # ────────────────  NN callback (from AI camera preview) ──────────────── #
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
            if self.enable_photo_save and (now - self.last_saved < MIN_INTERVAL_SEC):
                break # Respect min-interval only if saving photos
            
            # Get box coordinates relative to preview
            rel_box = tuple(boxes[i])
            x, y, w, h = self.imx.convert_inference_coords(
                rel_box, req.get_metadata(), self.cam
            )
            x1, y1 = x + w, y + h
            
            # Get preview dimensions for margin calculation
            w_pre, h_pre = self.preview_cfg["main"]["size"]
            x0_margin, y0_margin, x1_margin, y1_margin = add_margin(x, y, x1, y1, w_pre, h_pre)

            self.last_box = (x0_margin, y0_margin, x1_margin, y1_margin) # For overlay
            self.flash_frames = 5 # Trigger flash on preview

            if self.enable_photo_save:
                self.last_saved = now # Update last_saved time only if a save is triggered
                if self.capture_thread is None or not self.capture_thread.is_alive():
                    self.capture_thread = threading.Thread(
                        target=self._capture_hq_photo_worker,
                        daemon=True
                    )
                    self.capture_thread.start()
            break

    # ────────────────  HQ Photo Capture worker ──────────────── #
    def _capture_hq_photo_worker(self):
        time.sleep(0.05) # Allow callback to return
        ts = time.strftime("%Y%m%d_%H%M%S")

        if not self.enable_photo_save or not self.cam_hq:
            # Should not happen if logic in on_request is correct, but as a safeguard
            print("[INFO] HQ photo capture skipped (disabled or camera not available).")
            return

        try:
            print("[INFO] Capturing HQ photo with cam1...")
            self.cam_hq.start()
            hq_frame_preview = self.cam_hq.capture_array("main")
            hq_frame = cv2.cvtColor(hq_frame_preview, cv2.COLOR_BGR2RGB)
            self.cam_hq.stop()
            save_hq_photo(hq_frame, SAVE_DIR, ts)
        except Exception as e_hq:
            print(f"[WARN] HQ capture/save failed: {e_hq}")
            traceback.print_exc()
            if self.cam_hq: # Attempt to stop if error occurred after start
                try: self.cam_hq.stop()
                except: pass
        # AI camera (cam0) is not touched here; it continues its preview.

    # ────────────────  Main loop ──────────────── #
    def run(self):
        if self.enable_preview:
            print("[INFO] Live preview ON (from cam0) – press 'q' in window to quit.")
        else:
            print("[INFO] Live preview OFF – press Ctrl‑C to quit.")

        self.cam.start() # Start cam0 preview
        if self.enable_preview:
            cv2.namedWindow("Live Feed (cam0)", cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                frame_preview = self.cam.capture_array("main") # RGB preview from cam0
                display_bgr = frame_preview # OpenCV needs BGR

                if self.flash_frames > 0:
                    x0, y0, x1, y1 = self.last_box
                    cv2.rectangle(display_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.putText(
                        display_bgr, "Cat!", (x0, max(15, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                    self.flash_frames -= 1

                if self.enable_preview:
                    cv2.imshow("Live Feed (cam0)", display_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    time.sleep(0.01) # Yield CPU if no preview

        except KeyboardInterrupt:
            print("[INFO] Keyboard interrupt received.")
        finally:
            print("[INFO] Stopping cameras and cleaning up...")
            if self.cam: # AI Camera (cam0)
                try:
                    self.cam.stop()
                    self.cam.close()
                    print("[INFO] AI Camera (cam0) stopped and closed.")
                except Exception as e:
                    print(f"[WARN] Error stopping/closing AI camera (cam0): {e}")
            
            if self.cam_hq: # HQ Camera (cam1)
                try:
                    # Ensure it's stopped if it was running (e.g. error during capture)
                    self.cam_hq.stop() 
                    self.cam_hq.close()
                    print("[INFO] HQ Camera (cam1) stopped and closed.")
                except Exception: # Ignore if already stopped, not init, or other errors on close
                    pass 

            if self.enable_preview:
                cv2.destroyAllWindows()
            print("[INFO] Application terminated.")


# ──────────────────────────  CLI handling  ────────────────────────── #
def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cat detector with AI camera preview and optional HQ camera photo capture."
    )

    # Preview toggle (for AI camera) ---------------------------------- #
    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument(
        "--preview", dest="preview", action="store_true",
        help="Enable live preview window from AI camera (cam0) (default)."
    )
    preview_group.add_argument(
        "--no-preview", dest="preview", action="store_false",
        help="Disable live preview window."
    )
    parser.set_defaults(preview=True)

    # Photo saving toggle (for HQ camera) ----------------------------- #
    save_photo_group = parser.add_mutually_exclusive_group()
    save_photo_group.add_argument(
        "--save-photo", dest="save_photo", action="store_true",
        help="Enable capturing a photo with HQ camera (cam1) on detection (default)."
    )
    save_photo_group.add_argument(
        "--no-save-photo", dest="save_photo", action="store_false",
        help="Disable photo capture from HQ camera."
    )
    parser.set_defaults(save_photo=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    detector = None
    try:
        detector = CatDetector(enable_preview=args.preview,
                               enable_photo_save=args.save_photo)
        detector.run()
    except Exception as exc:
        print(f"[ERROR] Critical failure: {exc}")
        traceback.print_exc()
        if detector: # Attempt cleanup if detector object exists
            if detector.cam:
                try: detector.cam.stop()
                except: pass
                try: detector.cam.close()
                except: pass
            if detector.cam_hq:
                try: detector.cam_hq.stop()
                except: pass
                try: detector.cam_hq.close()
                except: pass
        sys.exit(1)