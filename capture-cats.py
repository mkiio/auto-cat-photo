#!/usr/bin/env python3
"""
Cat Detector with Live Preview & High-Res Crops (threaded)
----------------------------------------------------------
• Raspberry Pi 5 + Pi AI Camera (Sony IMX500), Picamera2 0.3+, OpenCV, PIL
• Shows live feed in an OpenCV window
• Detects cats (COCO ID 17) and flashes bounding box when captured
• Saves both rectangular crops (with 10% margin) and square 1024×1024 crops
  from full-resolution stills, without blocking the NN callback.
"""
from __future__ import annotations
import time, sys, traceback
import threading
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from picamera2 import Picamera2
from picamera2.devices import IMX500

# ─────────────────────────── User settings ─────────────────────────── #
CONF_THRESHOLD   = 0.60
MIN_INTERVAL_SEC = 10      # seconds between saves
MARGIN_RATIO     = 0.20    # 10% margin
CROP_SIZE        = 1024    # square crop size
SAVE_DIR         = Path("/home/mkiio/Developer/cat-detector/photos")
MODEL_BLOB       = (
    "/usr/share/imx500-models/"
    "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
)
COCO_CAT_ID = 17  # id for “cat” in COCO
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
    cx, cy = x0 + box_w//2, y0 + box_h//2
    half = side//2
    nx0 = max(0, cx - half); ny0 = max(0, cy - half)
    nx1 = min(w, nx0 + side); ny1 = min(h, ny0 + side)
    # adjust if clipped
    if nx1 - nx0 < side: nx0 = max(0, nx1 - side)
    if ny1 - ny0 < side: ny0 = max(0, ny1 - side)
    return nx0, ny0, nx1, ny1

def save_crops(rgb: np.ndarray, box, out_dir: Path):
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        # rectangular crop
        rect = rgb[y0:y1, x0:x1]
        Image.fromarray(rect).save(out_dir / f"cat_{ts}.jpg", quality=92)
        # square crop
        h_img, w_img, _ = rgb.shape
        sx0, sy0, sx1, sy1 = make_square_box(x0, y0, x1, y1, w_img, h_img)
        square = rgb[sy0:sy1, sx0:sx1]
        sq = Image.fromarray(square).resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
        sq.save(out_dir / f"cat_{ts}_square.jpg", quality=92)
        print(f"[INFO] saved cat_{ts}.jpg & cat_{ts}_square.jpg")
    except Exception as e:
        print(f"[WARN] failed to save crops: {e}")

class CatDetector:
    def __init__(self, model_blob: str = MODEL_BLOB):
        # NN + camera
        self.imx = IMX500(model_blob)
        self.cam = Picamera2(self.imx.camera_num)

        # preview (fast + inference)
        self.preview_cfg = self.cam.create_preview_configuration(
            main={"format":"RGB888","size":(640,480)}
        )
        # still (full-res)
        self.still_cfg   = self.cam.create_still_configuration(
            main={"format":"RGB888"}
        )
        self.cam.configure(self.preview_cfg)
        self.cam.post_callback = self.on_request

        # state
        self.last_saved    = 0.0
        self.flash_frames  = 0
        self.last_box      = (0,0,0,0)
        self.pending_box   = None      # preview coords awaiting high-res capture
        self.capture_thread = None

    def on_request(self, req):
        # NN inference
        try:
            outputs = self.imx.get_outputs(req.get_metadata())
        except Exception as e:
            print(f"[WARN] NN outputs read failed: {e}")
            return
        if not outputs or len(outputs) < 3:
            return
        boxes, scores, classes = outputs[:3]
        for i, score in enumerate(scores):
            if score < CONF_THRESHOLD: continue
            if int(classes[i]) != COCO_CAT_ID: continue
            now = time.time()
            if now - self.last_saved < MIN_INTERVAL_SEC:
                break
            # map to preview pixels
            rel = tuple(boxes[i])
            x, y, w, h = self.imx.convert_inference_coords(
                rel, req.get_metadata(), self.cam
            )
            x1, y1 = x+w, y+h
            preview = req.make_array("main")  # RGB @ 640×480
            h_pre, w_pre, _ = preview.shape
            x0, y0, x1, y1 = add_margin(x, y, x1, y1, w_pre, h_pre)

            # flash & record state
            self.last_saved   = now
            self.last_box     = (x0, y0, x1, y1)
            self.flash_frames = 5
            self.pending_box  = (x0, y0, x1, y1, w_pre, h_pre)

            # spawn a capture thread if none active
            if self.capture_thread is None or not self.capture_thread.is_alive():
                self.capture_thread = threading.Thread(
                    target=self._capture_highres_worker,
                    daemon=True
                )
                self.capture_thread.start()
            break

    def _capture_highres_worker(self):
        # give callback a moment to fully return & free locks
        time.sleep(0.05)
        try:
            # switch to still config
            self.cam.stop()
            self.cam.configure(self.still_cfg)
            self.cam.start()
            # grab full-res
            frame = self.cam.capture_array()  # BGR888 @ full sensor
            full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_full, w_full, _ = full.shape

            # scale the preview box into full-res coords
            x0, y0, x1, y1, w_pre, h_pre = self.pending_box
            sx, sy = w_full / w_pre, h_full / h_pre
            box_full = (
                int(x0*sx), int(y0*sy),
                int(x1*sx), int(y1*sy),
            )
            save_crops(full, box_full, SAVE_DIR)

        except Exception as e:
            print(f"[ERROR] high-res capture failed: {e}")
        finally:
            # back to preview
            try:
                self.cam.stop()
                self.cam.configure(self.preview_cfg)
                self.cam.start()
            except Exception:
                pass

    def run(self):
        print("[INFO] starting live preview. Press 'q' to quit.")
        self.cam.start()
        cv2.namedWindow('Live Feed', cv2.WINDOW_AUTOSIZE)
        try:
            while True:
                frame = self.cam.capture_array()    # RGB preview
                bgr   = frame

                # flash box
                if self.flash_frames > 0:
                    x0,y0,x1,y1 = self.last_box
                    cv2.rectangle(bgr,(x0,y0),(x1,y1),(0,255,0),2)
                    cv2.putText(
                        bgr, "Cat!", (x0, max(15,y0-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2
                    )
                    self.flash_frames -= 1

                cv2.imshow('Live Feed', bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.01)

        except KeyboardInterrupt:
            pass
        finally:
            self.cam.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        CatDetector().run()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
