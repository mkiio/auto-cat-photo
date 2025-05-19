# capture_controller.py
import time
import logging
from pathlib import Path
import threading
from typing import Optional, Tuple, Any, Callable
import numpy as np

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
        def capture_array(self, stream_name="main"): print(f"MockPicamera2 (HQ) capturing array from {stream_name}."); return np.zeros((1080, 1920, 3), dtype=np.uint8) # Mock HQ size
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
# Assuming CatDetectedEventData is defined in detection_controller or a shared types file
# For now, let's copy/define it if not using a shared types module.
# from detection_controller import CatDetectedEventData # This would create a circular dependency if not careful.
# Instead, let's define the expected structure or use Any.
from typing import Any, Tuple, Optional
CatDetectedEventData = Any # Or define as Tuple[Optional[np.ndarray], Tuple[int, int, int, int], float, str]


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
        self.hq_format = self.config.get('hq_camera.format')
        self.hq_jpeg_quality = self.config.get('hq_camera.jpeg_quality')
        self.save_dir = self.config.SAVE_DIR_ABSOLUTE # Use resolved absolute path
        self.min_interval_sec = self.config.get('detection.min_interval_sec')

        self.cam_hq: Optional[Picamera2] = None
        self.still_cfg_hq: Optional[dict] = None

        self.last_saved_time: float = 0.0
        self.capture_thread: Optional[threading.Thread] = None
        self.is_hq_camera_initialized = False
        self._lock = threading.Lock() # For thread safety around camera ops and last_saved_time

    def _initialize_hq_camera(self):
        if not self.enable_photo_save:
            self.logger.info("Photo saving is disabled. HQ Camera will not be initialized.")
            return False

        if not PICAMERA2_AVAILABLE:
            self.logger.warning("Picamera2 library not found. Using MOCK HQ camera. Photo saving will be simulated.")

        try:
            self.logger.info(f"Initializing HQ Camera (cam{self.hq_camera_num})...")
            self.cam_hq = Picamera2(self.hq_camera_num) # type: ignore
            # main={"format": self.hq_format} # Original script used RGB888 for PIL
            # Picamera2 can save JPEGs directly, which might be more efficient if not needing array first.
            # However, to stick to original logic of getting array then saving with PIL:
            main_config = {"format": self.hq_format}
            capture_res = self.config.get('hq_camera.capture_resolution', None)
            if capture_res and len(capture_res) == 2:
                main_config["size"] = tuple(capture_res)
                self.logger.info(f"HQ Camera resolution set to: {main_config['size']}")
            else:
                self.logger.info("HQ Camera resolution set to maximum.")


            self.still_cfg_hq = self.cam_hq.create_still_configuration(main=main_config) # type: ignore
            self.cam_hq.configure(self.still_cfg_hq) # type: ignore
            self.is_hq_camera_initialized = True
            self.logger.info(f"HQ Camera (cam{self.hq_camera_num}) initialized successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize HQ Camera (cam{self.hq_camera_num}): {e}. Disabling photo capture.", exc_info=True)
            self.cam_hq = None
            self.is_hq_camera_initialized = False
            self.enable_photo_save = False # Auto-disable if init fails
            return False

    def handle_cat_detected(self, event_data: CatDetectedEventData):
        """
        Callback function to be triggered by DetectionController when a cat is detected.
        event_data might be: (frame_array, box_coords, score, class_name)
        """
        if not self.enable_photo_save or not self.is_hq_camera_initialized or not self.cam_hq:
            self.logger.debug("Photo saving disabled or HQ camera not ready. Skipping capture.")
            return

        with self._lock: # Ensure thread safety for time check and thread start
            now = time.time()
            if (now - self.last_saved_time) < self.min_interval_sec:
                self.logger.info(f"Cat detected, but still within min interval ({self.min_interval_sec}s). Skipping HQ capture.")
                return

            if self.capture_thread and self.capture_thread.is_alive():
                self.logger.warning("HQ capture requested, but a previous capture thread is still active. Skipping.")
                return

            self.logger.info("Cat detected event received. Conditions met for HQ capture.")
            self.last_saved_time = now # Update time immediately to prevent rapid re-triggering

            self.capture_thread = threading.Thread(target=self._capture_and_save_hq_photo_worker, daemon=True)
            self.capture_thread.start()

    def _capture_and_save_hq_photo_worker(self):
        """Worker thread to perform the actual HQ capture and saving."""
        if not self.cam_hq: # Should be checked before starting thread, but double check
            self.logger.error("HQ camera not available in worker thread.")
            return

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"HQ Capture Worker: Starting capture for timestamp {timestamp_str}...")

        try:
            # Picamera2 needs to be started before capture and stopped after for stills
            # if not continuously running.
            self.logger.debug("HQ Capture Worker: Starting HQ camera for single capture...")
            self.cam_hq.start() # Start the camera for capture
            time.sleep(0.1) # Brief pause for camera to be ready, adjust if needed

            self.logger.debug("HQ Capture Worker: Capturing HQ frame array...")
            # This captures into a NumPy array in RGB888 format (or as configured)
            hq_frame_array = self.cam_hq.capture_array("main") # type: ignore
            self.logger.info(f"HQ Capture Worker: Frame captured with shape {hq_frame_array.shape} and type {hq_frame_array.dtype}.")

            # Original script converted BGR to RGB, but if format is already RGB888, this might not be needed
            # If hq_format is 'BGR888', then conversion is needed for PIL.
            # If hq_format is 'RGB888', no conversion needed if PIL handles it.
            # Let's assume hq_format is RGB888 as per original.
            # No conversion needed: rgb_to_save = hq_frame_array
            # If it were BGR:
            # import cv2
            # rgb_to_save = cv2.cvtColor(hq_frame_array, cv2.COLOR_BGR2RGB)

            rgb_to_save = hq_frame_array # Assuming format is already RGB for PIL

            self.logger.debug("HQ Capture Worker: Stopping HQ camera after capture.")
            self.cam_hq.stop() # Stop the camera

            self._save_photo_to_disk(rgb_to_save, timestamp_str)

        except Exception as e:
            self.logger.error(f"HQ Capture Worker: Error during HQ capture or save: {e}", exc_info=True)
            if self.cam_hq: # Attempt to stop if error occurred after start
                try:
                    self.logger.debug("HQ Capture Worker: Attempting to stop HQ camera due to error.")
                    self.cam_hq.stop()
                except Exception as e_stop:
                    self.logger.error(f"HQ Capture Worker: Error stopping HQ camera after failure: {e_stop}", exc_info=True)
        finally:
            self.logger.info(f"HQ Capture Worker for timestamp {timestamp_str} finished.")


    def _save_photo_to_disk(self, image_array: np.ndarray, timestamp: str):
        """Saves the captured image array to disk."""
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.save_dir / f"cat_{timestamp}_HQ.jpg"
            self.logger.info(f"Saving HQ photo to: {filepath}")

            # Use Pillow to save the NumPy array as JPEG
            pil_image = Image.fromarray(image_array) # type: ignore
            pil_image.save(filepath, quality=self.hq_jpeg_quality)

            self.logger.info(f"Successfully saved HQ photo: {filepath.name}")
        except Exception as e:
            self.logger.error(f"Failed to save HQ photo '{filepath_name if 'filepath_name' in locals() else 'unknown'}': {e}", exc_info=True)


    def start(self):
        """Initializes the HQ camera if photo saving is enabled."""
        self.logger.info("CaptureController starting...")
        if self.enable_photo_save:
            if not self._initialize_hq_camera():
                self.logger.warning("CaptureController started, but HQ camera failed to initialize. Photo capture disabled.")
            else:
                self.logger.info("CaptureController started, HQ camera initialized and ready for captures.")
        else:
            self.logger.info("CaptureController started, photo saving is disabled.")

    def stop(self):
        """Cleans up resources, like closing the HQ camera."""
        self.logger.info("CaptureController stopping...")
        if self.capture_thread and self.capture_thread.is_alive():
            self.logger.info("Waiting for ongoing HQ capture thread to finish...")
            self.capture_thread.join(timeout=5.0) # Wait for a bit
            if self.capture_thread.is_alive():
                self.logger.warning("Capture thread did not finish in time.")

        if self.cam_hq:
            try:
                self.logger.info("Stopping and closing HQ Camera...")
                # Ensure camera is stopped if it was started for a capture and an error occurred
                self.cam_hq.stop() # Safe to call even if already stopped
                self.cam_hq.close()
                self.logger.info("HQ Camera stopped and closed.")
            except Exception as e:
                self.logger.error(f"Error stopping/closing HQ camera: {e}", exc_info=True)
        self.is_hq_camera_initialized = False
        self.cam_hq = None
        self.logger.info("CaptureController stopped.")


if __name__ == '__main__':
    # Example Usage
    class DummyArgs: preview = None; save_photo = None; save_dir = None; conf_threshold = None; min_interval = None
    cfg = Config(args=DummyArgs()) # Ensure config.py is accessible
    logger = setup_logging(cfg)    # Ensure logger_setup.py is accessible

    if not cfg.get('application.save_photos'):
        logger.info("Photo saving is disabled in config. CaptureController example will be limited.")
        # exit() # We can still test initialization failure or disabled state

    capture_ctrl = CaptureController(config=cfg)
    capture_ctrl.start() # Initializes the HQ camera

    if cfg.get('application.save_photos') and capture_ctrl.is_hq_camera_initialized:
        logger.info("Simulating a cat detection event to trigger capture...")
        # Dummy event data (frame part is not used by current handle_cat_detected)
        dummy_event_data: CatDetectedEventData = (None, (10,10,100,100), 0.95, "cat")
        capture_ctrl.handle_cat_detected(dummy_event_data)

        # Wait for the thread to likely complete for this example
        time.sleep(5) # Give some time for the capture thread to run

        logger.info("Simulating another detection event soon after (should be skipped if interval is too short)...")
        time.sleep(2) # Shorter than typical min_interval
        capture_ctrl.handle_cat_detected(dummy_event_data)
        time.sleep(1)


        logger.info(f"Simulating another detection event after {cfg.get('detection.min_interval_sec')}s...")
        time.sleep(cfg.get('detection.min_interval_sec'))
        capture_ctrl.handle_cat_detected(dummy_event_data)
        time.sleep(5)


    else:
        logger.info("Photo saving is disabled or HQ camera failed to initialize. No capture will be triggered.")

    capture_ctrl.stop()
    logger.info("CaptureController example finished.")