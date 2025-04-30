# Auto Cat Photo Capture for Raspberry Pi

Cat detection and auto photo capture using the Raspberry Pi AI Camera (Sony IMX500) On-sensor inference.  Can save full frame or rectangular or square crops of the cat with optional live preview.
---

## ✨ Key features

| Feature | Details |
|---------|---------|
| **On-chip inference** | Uses the IMX500’s built-in MobileNet-SSD COCO model for real-time detection. |
| **Live preview** | Previe wwindow at 640×480; disable with `--no-preview`. |
| **High-resolution full frame or crops** | Switches to full-sensor mode in a background thread; saves:<br>• `cat_YYYYMMDD_HHMMSS.jpg` |
| **Rate limiting** | Configurable delay between captures (s)|
| **Config toggles** | Threshold, margin, crop size, save interval, save folder are one-line constants at the top. |
| **Thread-safe** | Detection and high-res capture happen in separate threads so inference is never blocked. |

---

## 🖥️ Tested hardware & OS

* Raspberry Pi 5 (4+ GB) w/Raspberry Pi OS
* Raspberry Pi AI Camera 
* Python 3.11 + Picamera2 ≥ 0.3

---

## 📦 Dependencies

| Package | Version (tested) |
|---------|------------------|
| `python3-picamera2` | 0.3.15 |
| `python3-opencv`    | 4.10.* |
| `python3-pil` (Pillow) | 10.* |
| `numpy`             | 1.26.* |
| `imx500-all` & `imx500-tools` | (latest) |

Install any missing Python deps with:

```
sudo apt install python3-picamera2 python3-opencv python3-pil
```

---

## 🤖 Neural-network model

* Model: MobileNet-SSD (COCO, 320 × 320)
* Class ID for cat: 17
* Blob path: /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk

## 📂 Where photos are saved

By default the script saves into a folder photos/ next to the script file:
```
SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_DIR   = SCRIPT_DIR / "photos"
```
One JPEGs are written per detection. Quality is set to 92. The folder is auto-created on first run.

---

## 🚀 Usage
```
# Default: preview ON, square
python3 capture_cats.py

# Headless capture (no OpenCV window)
python3 capture_cats.py --no-preview

# Headless capture (no OpenCV window)
python3 capture_cats.py --capture {square|crop|full} 
```

Command-line flags


| Flag | Effect |
|------|--------|
| `--preview`     | Show live OpenCV window (default). |
| `--no-preview`  | Disable window; quit with **Ctrl-C**. |
| `--capture {square|crop|full}` | Change capture format |

## ⚙️ Tunable constants (top of file)

| Constant | Default | Purpose |
|----------|---------|---------|
| `CONF_THRESHOLD`   | `0.60` | Detection confidence threshold |
| `MIN_INTERVAL_SEC` | `10`   | Minimum seconds between saves |
| `MARGIN_RATIO`     | `0.20` | 10 % margin around bbox (0.20 = ±10 %) |
| `CROP_SIZE`        | `1024` | Square crop side length |
| `SAVE_DIR`         | see above | Folder for saved images |

---

## MIT License

Copyright (c) 2025 mkiio

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
