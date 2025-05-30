# Auto Cat Photo 2

<img alt="Raspberry Pi with Pi AI and Pi HQ Camera on a tripod" src="https://github.com/mkiio/auto-cat-photo/blob/main/autocampi.jpeg" width="256" />


**Automatically detect and capture photos of cats using a Raspberry Pi**

Auto Cat Photo 2 uses a the Pi AI camera for efficient cat detection and a Pi HQ camera for capturing images. 

## How It Works

1.  **Detection Camera: Raspberry Pi AI Camera**: This camera provides a continuous video feed that is processed by an onboard AI model (SSD MobileNetV2) to detect the presence of cats.
2.  **High-Quality Capture Camera: Raspberry Pi HQ Camera**: When the AI camera detects a cat with sufficient confidence, this second camera is triggered to capture a high-resolution photograph.

Key features include:
* **Live Preview**: Optionally, display a live feed from the AI camera with detected cats highlighted.
* **Configurable Detection**: Adjust detection confidence, minimum interval between captures, and other parameters.
* **Robust Logging**: Detailed logs for monitoring and troubleshooting.
* **External Configuration**: Settings are managed via a `settings.yaml` file, allowing for easy adjustments without code changes.

## Hardware Requirements

* **Raspberry Pi**:
    * Raspberry Pi 5 (4GB RAM or more recommended) 
    * Adequate power supply for the Raspberry Pi and connected peripherals.
* **Cameras**:
    * **Detection Camera**:
        *  Raspberry Pi AI Camera w/Sony IMX500 based AI camera 
    * **Capture Camera**:
        * Raspberry Pi HQ Camera(Sony IMX477 sensor) or Raspberry Pi Camera Module 2/3.

## Software & Dependencies

* **Libraries**:
    * `picamera2`: For camera control on Raspberry Pi.
    * `PyYAML`: For loading an external configuration file.
    * `numpy`: For numerical operations, especially image array manipulation.
    * `opencv-python` (`cv2`): For image processing and optional live preview display.
    * `Pillow` (`PIL`): For image saving.

## Quick Start

1.  **Prepare your Raspberry Pi**:
    * Ensure your Raspberry Pi OS is up to date:
        ```bash
        sudo apt update
        sudo apt full-upgrade -y
        ```
    * Enable camera interfaces:
        ```bash
        sudo raspi-config
        ```
        Navigate to `Interface Options` -> `Camera` and enable it. You might also need to enable I2C/SPI depending on your specific AI camera module if it uses them. Reboot if prompted.
    * Ensure `libcamera` and `picamera2` are installed. `picamera2` is typically pre-installed on recent Raspberry Pi OS images. If not:
        ```bash
        sudo apt install -y python3-picamera2
        ```

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/mkiio/auto-cat-photo
    cd auto-cat-photo
    ```

3.  **Install Python Dependencies**:
    It's recommended to use a Python virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

4.  **Configure the Application**:
    * Copy or rename the example configuration file if one is provided, or create `settings.yaml` based on the structure shown in `config.py` or the example in the previous response.
        ```yaml
        # settings.yaml
        ai_camera:
          camera_num: 0
          preview_resolution: [640, 480]
          format: "RGB888"
        hq_camera:
          camera_num: 1
          capture_resolution: null # null for max, or e.g., [1920, 1080]
          format: "RGB888"
          jpeg_quality: 95
        detection:
          model_blob_path: "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk" # Adjust if your model is elsewhere or different
          coco_cat_id: 17
          confidence_threshold: 0.50
          min_interval_sec: 10
          preview_box_margin_ratio: 0.20
        application:
          save_photos: true
          save_dir: "photos"
          enable_preview: true
        logging:
          log_file: "cat_detector.log"
          log_level: "INFO"
          log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
          log_to_console: true
          log_to_file: true
        ```
    * **Important**: Verify the `model_blob_path` in `settings.yaml`. This path should point to the AI model file used for detection. If your model is located elsewhere, update this path.
    * Adjust `camera_num` for `ai_camera` and `hq_camera` if your cameras are not detected as `0` and `1` respectively. You can list available cameras using `libcamera-hello --list-cameras`.

5.  **Run the Application**:

    ```bash
    python app.py
    ```
    * Photos will be saved in the directory specified by `save_dir` in `settings.yaml` (default is `photos/` in the application directory).
    * Logs will be created as `cat_detector.log` (by default).
    * If preview is enabled, a window will appear showing the AI camera feed. Press 'q' in this window to quit.
    * Press `Ctrl+C` in the terminal to stop the application gracefully.

6.  **Command-Line Overrides**:
    You can override some settings from `settings.yaml` using command-line arguments:
    ```bash
    python app.py --no-preview --save-photo --save-dir /media/my_usb/cat_captures --conf-threshold 0.60
    ```
    Run `python app.py --help` for a full list of available command-line arguments.

## Troubleshooting

* **Camera Not Detected**:
    * Ensure cameras are properly connected and enabled in `raspi-config`.
    * Use `libcamera-hello --list-cameras` to check if the system recognizes your cameras and their assigned numbers. Adjust `camera_num` in `settings.yaml` accordingly.
    * Check `dmesg` for any hardware-related error messages.
* **Model File Not Found**: Double-check the `model_blob_path` in `settings.yaml`.
* **Permission Errors**: Ensure the application has write permissions to the `save_dir` and the directory for `log_file`.
* **Performance Issues**:
    * Ensure your Raspberry Pi has adequate cooling.
    * If not using a dedicated AI camera, detection on the CPU can be intensive. You might need to adjust preview resolution or AI model complexity.
* **Check Logs**: The application logs to `cat_detector.log` (by default) and the console. These logs provide valuable information for diagnosing issues.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
