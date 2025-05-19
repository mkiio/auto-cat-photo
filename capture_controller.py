import time
import logging
from pathlib import Path
import threading
import numpy as np
import cv2 # <--- ADD THIS IMPORT FOR cv2.cvtColor

# Conditional import for Picamera2 to allow for testing on non-Pi environments
try:
    from picamera2 import Picamera2
    from PIL import Image # Pillow for saving
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    # Mock classes for testing
    class Picamera2Mock:
        def __init__(self, camera_num=0): self.camera_num = camera_num; self._is_running = False; self._config = None; print(f"MockPicamera2(cam={camera_num}) created for HQ.")
        def create_still_configuration(self, main=None, lores=None, raw=None, transform=None, colour_space=None, buffer_count=1, controls=None, display=None, encode=None): self._config = {"main": main}; return self._config
        def configure(self, cfg): print(f"MockPicamera2 (HQ) configured with: {cfg}")
        def start(self): self._is_running = True; print("MockPicamera2 (HQ) started.")
        def stop(self): self._is_running = False; print("MockPicamera2 (HQ) stopped.")
        def capture_array(self, stream_name="main"): print(f"MockPicamera2 (HQ) capturing array from {stream_name}."); return np.zeros((1080, 1920, 3), dtype=np.uint8)
        def close(self): print("MockPicamera2 (HQ) closed.")

    class ImageMock:
        @staticmethod
        def fromarray(array_obj):
            print(f"MockImage.fromarray called with array of shape {array_obj.shape}")
            class SavableMockImage:
                def save(self, filepath, quality=95):
                    print(f"MockImage.save called for {filepath} with quality {quality}")
            return SavableMockImage()

    Picamera2 = Picamera2Mock # type: ignore
    Image = ImageMock # type: ignore


from config import Config
from typing import Any, Tuple, Optional # Ensure typing imports are present
# CatDetectedEventData definition might be here or imported if shared
CatDetectedEventData = Any


class CaptureController:
    """
    Manages the HQ Camera (cam1) for capturing photos upon cat detection.
    Subscribes to 'cat_detected_event', enforces capture interval, saves images.
    """
    def __init__(self, config: Config):
        self.logger = logging.getLogger("CatDetectorApp.CaptureController")
        self.config = config

        self.enable_photo_save = self.config.get('application.save_photos')
        self.hq_camera_num = self.config.get('hq_camera.camera_num')
        # It's good to ensure hq_format from config is what you expect (e.g., "BGR888" or "RGB888")
        # For this fix, we'll assume capture_array gives BGR.
        self.hq_format = self.config.get('hq_camera.format', "BGR888") # Default to BGR888 if not specified
        self.hq_jpeg_quality = self.config.get('hq_camera.jpeg_quality')
        self.save_dir = self.config.SAVE_DIR_ABSOLUTE
        self.min_interval_sec = self.config.get('detection.min_interval_sec')

        self.cam_hq: Optional[Picamera2] = None
        self.still_cfg_hq: Optional[dict] = None

        self.last_saved_time: float = 0.0
        self.capture_thread: Optional[threading.Thread] = None
        self.is_hq_camera_initialized = False
        self._lock = threading.Lock()

    def _initialize_hq_camera(self):
        if not self.enable_photo_save:
            self.logger.info("Photo saving is disabled. HQ Camera will not be initialized.")
            return False

        if not PICAMERA2_AVAILABLE:
            self.logger.warning("Picamera2 library not found. Using MOCK HQ camera. Photo saving will be simulated.")

        try:
            self.logger.info(f"Initializing HQ Camera (cam{self.hq_camera_num})...")
            self.cam_hq = Picamera2(self.hq_camera_num) # type: ignore
            
            main_config = {"format": self.hq_format} # Use the format from config
            capture_res = self.config.get('hq_camera.capture_resolution', None)
            if capture_res and len(capture_res) == 2:
                main_config["size"] = tuple(capture_res)
                self.logger.info(f"HQ Camera resolution set to: {main_config['size']}")
            else:
                self.logger.info("HQ Camera resolution set to maximum for format.")

            self.still_cfg_hq = self.cam_hq.create_still_configuration(main=main_config) # type: ignore
            self.cam_hq.configure(self.still_cfg_hq) # type: ignore
            self.is_hq_camera_initialized = True
            self.logger.info(f"HQ Camera (cam{self.hq_camera_num}) initialized successfully with format {self.hq_format}.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize HQ Camera (cam{self.hq_camera_num}): {e}. Disabling photo capture.", exc_info=True)
            self.cam_hq = None
            self.is_hq_camera_initialized = False
            self.enable_photo_save = False
            return False

    def handle_cat_detected(self, event_data: CatDetectedEventData):
        if not self.enable_photo_save or not self.is_hq_camera_initialized or not self.cam_hq:
            self.logger.debug("Photo saving disabled or HQ camera not ready. Skipping capture.")
            return

        with self._lock:
            now = time.time()
            if (now - self.last_saved_time) < self.min_interval_sec:
                self.logger.info(f"Cat detected, but still within min interval ({self.min_interval_sec}s). Skipping HQ capture.")
                return

            if self.capture_thread and self.capture_thread.is_alive():
                self.logger.warning("HQ capture requested, but a previous capture thread is still active. Skipping.")
                return

            self.logger.info("Cat detected event received. Conditions met for HQ capture.")
            self.last_saved_time = now
            self.capture_thread = threading.Thread(target=self._capture_and_save_hq_photo_worker, daemon=True)
            self.capture_thread.start()

    def _capture_and_save_hq_photo_worker(self):
        if not self.cam_hq:
            self.logger.error("HQ camera not available in worker thread.")
            return

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"HQ Capture Worker: Starting capture for timestamp {timestamp_str}...")

        try:
            self.logger.debug("HQ Capture Worker: Starting HQ camera for single capture...")
            self.cam_hq.start()
            time.sleep(0.2) # Increased sleep slightly for camera to stabilize, adjust if needed

            self.logger.debug("HQ Capture Worker: Capturing HQ frame array...")
            # Assuming hq_frame_array from Picamera2 is in BGR order
            hq_frame_array_bgr = self.cam_hq.capture_array("main") # type: ignore
            self.logger.info(f"HQ Capture Worker: Frame captured with shape {hq_frame_array_bgr.shape} and type {hq_frame_array_bgr.dtype}.")

            # Convert BGR to RGB for Pillow (PIL)
            self.logger.debug("HQ Capture Worker: Converting BGR frame to RGB for saving.")
            rgb_to_save = cv2.cvtColor(hq_frame_array_bgr, cv2.COLOR_BGR2RGB)

            self.logger.debug("HQ Capture Worker: Stopping HQ camera after capture.")
            self.cam_hq.stop()

            self._save_photo_to_disk(rgb_to_save, timestamp_str)
        except Exception as e:
            self.logger.error(f"HQ Capture Worker: Error during HQ capture or save: {e}", exc_info=True)
            if self.cam_hq:
                try:
                    self.logger.debug("HQ Capture Worker: Attempting to stop HQ camera due to error.")
                    self.cam_hq.stop()
                except Exception as e_stop:
                    self.logger.error(f"HQ Capture Worker: Error stopping HQ camera after failure: {e_stop}", exc_info=True)
        finally:
            self.logger.info(f"HQ Capture Worker for timestamp {timestamp_str} finished.")


    def _save_photo_to_disk(self, image_array_rgb: np.ndarray, timestamp: str):
        """Saves the captured RGB image array to disk."""
        filepath = self.save_dir / f"cat_{timestamp}_HQ.jpg" # Define here for logging
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Saving HQ photo to: {filepath}")

            pil_image = Image.fromarray(image_array_rgb) # type: ignore
            pil_image.save(filepath, quality=self.hq_jpeg_quality)
            self.logger.info(f"Successfully saved HQ photo: {filepath.name}")
        except Exception as e:
            self.logger.error(f"Failed to save HQ photo '{filepath.name}': {e}", exc_info=True)

    def start(self):
        self.logger.info("CaptureController starting...")
        if self.enable_photo_save:
            if not self._initialize_hq_camera():
                self.logger.warning("CaptureController started, but HQ camera failed to initialize. Photo capture disabled.")
            else:
                self.logger.info("CaptureController started, HQ camera initialized and ready for captures.")
        else:
            self.logger.info("CaptureController started, photo saving is disabled.")

    def stop(self):
        self.logger.info("CaptureController stopping...")
        if self.capture_thread and self.capture_thread.is_alive():
            self.logger.info("Waiting for ongoing HQ capture thread to finish...")
            self.capture_thread.join(timeout=5.0)
            if self.capture_thread.is_alive():
                self.logger.warning("Capture thread did not finish in time.")

        if self.cam_hq:
            try:
                self.logger.info("Stopping and closing HQ Camera...")
                self.cam_hq.stop()
                self.cam_hq.close()
                self.logger.info("HQ Camera stopped and closed.")
            except Exception as e:
                self.logger.error(f"Error stopping/closing HQ camera: {e}", exc_info=True)
        self.is_hq_camera_initialized = False
        self.cam_hq = None
        self.logger.info("CaptureController stopped.")

if __name__ == '__main__':
    class DummyArgsForConfig: preview = None; save_photo = None; save_dir = None; conf_threshold = None; min_interval = None
    
    # Basic logger for test if full setup fails
    try:
        from logger_setup import setup_logging as app_setup_logging # Alias to avoid conflict
        cfg_test = Config(args=DummyArgsForConfig())
        logger_test = app_setup_logging(cfg_test)
    except Exception:
        import logging as default_logging # Alias
        default_logging.basicConfig(level=default_logging.INFO)
        logger_test = default_logging.getLogger("CaptureControllerTest")
        logger_test.info("Using basic logging for CaptureController test.")
        # Create a minimal config for testing if full config load fails
        class MinimalConfigTest:
            SAVE_DIR_ABSOLUTE = Path("./photos_test")
            def get(self, key, default=None):
                vals = {
                    'application.save_photos': True, 'hq_camera.camera_num': 1,
                    'hq_camera.format': "BGR888", 'hq_camera.jpeg_quality': 90,
                    'detection.min_interval_sec': 5
                }
                return vals.get(key, default)
        cfg_test = MinimalConfigTest()


    if not cfg_test.get('application.save_photos'):
        logger_test.info("Photo saving is disabled in config. CaptureController example will be limited.")
    
    capture_ctrl_test = CaptureController(config=cfg_test)
    capture_ctrl_test.start()

    if cfg_test.get('application.save_photos') and capture_ctrl_test.is_hq_camera_initialized:
        logger_test.info("Simulating a cat detection event to trigger capture...")
        dummy_event_data_test: CatDetectedEventData = (None, (10,10,100,100), 0.95, "cat")
        capture_ctrl_test.handle_cat_detected(dummy_event_data_test)
        time.sleep(5)
        logger_test.info(f"Simulating another detection event after {cfg_test.get('detection.min_interval_sec')}s...")
        time.sleep(cfg_test.get('detection.min_interval_sec'))
        capture_ctrl_test.handle_cat_detected(dummy_event_data_test)
        time.sleep(5)
    else:
        logger_test.info("Photo saving is disabled or HQ camera failed to initialize. No capture will be triggered.")

    capture_ctrl_test.stop()
    logger_test.info("CaptureController example finished.")