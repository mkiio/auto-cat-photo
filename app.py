# app.py
# Ensure these imports are at the top
import time
import signal
import sys
import logging
import threading
from typing import Optional, Tuple, Any, Callable

import numpy as np # Make sure numpy is imported as np
import cv2 # OpenCV for display

from config import Config, ConfigError, parse_cli_args
from logger_setup import setup_logging
from detection_controller import DetectionController, CatDetectedEventData
from capture_controller import CaptureController

OPENCV_AVAILABLE = True # Assuming cv2 imported successfully for this snippet

class Application:
    """
    Main application orchestrator.
    Initializes all modules, manages the main lifecycle, and coordinates interactions.
    """
    def __init__(self):
        self.config: Optional[Config] = None
        self.logger: Optional[logging.Logger] = None
        self.detection_controller: Optional[DetectionController] = None
        self.capture_controller: Optional[CaptureController] = None

        self._running = threading.Event()
        self._preview_enabled = False
        self._flash_frames_preview = 0
        self._last_box_preview_cv: Optional[Tuple[int,int,int,int]] = None


    def _setup_signal_handlers(self):
        """Sets up signal handlers for graceful shutdown."""
        if self.logger: # Ensure logger is initialized
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
            self.logger.info("Signal handlers for SIGINT and SIGTERM registered.")
        else:
            # Fallback if logger is not ready (should not happen in normal flow)
            print("Logger not available for signal handler setup.", file=sys.stderr)


    def _handle_signal(self, signum, frame):
        if self.logger:
            self.logger.warning(f"Signal {signal.Signals(signum).name} received. Initiating graceful shutdown...")
        else:
            print(f"Signal {signal.Signals(signum).name} received. Initiating graceful shutdown...", file=sys.stderr)
        self.stop()

    def _initialize_modules(self):
        """Initializes all application components."""
        try:
            cli_args = parse_cli_args()
            self.config = Config(args=cli_args)
            self.logger = setup_logging(self.config)
        except ConfigError as e:
            print(f"[CRITICAL] Configuration error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"[CRITICAL] Failed during initial setup (config/logging): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

        self.logger.info("Application configuration and logging initialized.")

        try:
            self.detection_controller = DetectionController(self.config)
            self.capture_controller = CaptureController(self.config)

            self.detection_controller.subscribe_to_cat_detections(
                self.capture_controller.handle_cat_detected
            )
            self.logger.info("Subscribed CaptureController to cat detection events.")

            self._preview_enabled = self.config.get('application.enable_preview')
            if self._preview_enabled:
                if not OPENCV_AVAILABLE: # cv2 should be available if we reach here
                    self.logger.warning("OpenCV is not available, cannot enable live preview window.")
                    self._preview_enabled = False
                else:
                    self.logger.info("Live preview enabled. Setting up display callback.")
                    self.detection_controller.subscribe_to_frame_ready(self._display_preview_frame)
                    cv2.namedWindow("Cat Detector - AI Feed (cam0)", cv2.WINDOW_AUTOSIZE)
        except Exception as e:
            self.logger.critical(f"Failed to initialize application controllers or setup communication: {e}", exc_info=True)
            sys.exit(1)
        self.logger.info("All application modules initialized successfully.")


    def _display_preview_frame(self, frame: Optional[np.ndarray], detection_box: Optional[Tuple[int, int, int, int]]):
        """
        Callback for DetectionController to display frames via OpenCV.
        Assumes 'frame' from Picamera2's capture_array is in BGR order.
        """
        if not self._preview_enabled or frame is None:
            return

        try:
            # Frame from Picamera2 capture_array is assumed to be BGR.
            # OpenCV's imshow expects BGR, so no conversion is needed here.
            display_image = frame # Use frame directly

            if detection_box:
                self._last_box_preview_cv = detection_box
                self._flash_frames_preview = 5
                self.logger.debug(f"Preview: New detection box received: {detection_box}")

            if self._flash_frames_preview > 0 and self._last_box_preview_cv:
                x0, y0, x1, y1 = self._last_box_preview_cv
                cv2.rectangle(display_image, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(
                    display_image, "Cat!", (x0, max(15, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                self._flash_frames_preview -= 1

            cv2.imshow("Cat Detector - AI Feed (cam0)", display_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.logger.info("Preview window: 'q' pressed. Initiating shutdown.")
                self.stop()
            elif key == ord('s'):
                self.logger.info("Preview window: 's' pressed. Manually triggering cat detected event for HQ capture.")
                if self.capture_controller and self.detection_controller:
                    dummy_event_data: CatDetectedEventData = (None, (10,10,100,100), 0.99, "manual_cat_trigger") # type: ignore
                    self.capture_controller.handle_cat_detected(dummy_event_data)
        except Exception as e:
            self.logger.error(f"Error in preview display callback: {e}", exc_info=True)


    def run(self):
        """Main application entry point."""
        self._initialize_modules()
        self._setup_signal_handlers()
        self._running.set()

        self.logger.info("Starting application components...")
        try:
            if self.capture_controller:
                self.capture_controller.start()
            if self.detection_controller:
                self.detection_controller.start_detection()

            self.logger.info("Application is now running. Press Ctrl+C to exit.")
            if self._preview_enabled:
                self.logger.info("Live preview is ON. Press 'q' in the preview window to quit.")
            else:
                self.logger.info("Live preview is OFF.")

            if self.detection_controller and self._preview_enabled:
                self.detection_controller.run_preview_loop_if_needed()
            else:
                while self._running.is_set():
                    time.sleep(0.5)
        except RuntimeError as e:
            self.logger.critical(f"Runtime error during application execution: {e}", exc_info=True)
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt in main run loop. Shutting down.")
        except Exception as e:
            self.logger.critical(f"Unhandled exception in main run loop: {e}", exc_info=True)
        finally:
            if self._running.is_set():
                self.logger.info("Main loop ended, ensuring graceful shutdown.")
                self.stop()
            else:
                self.logger.info("Application has already been signaled to stop.")

    def stop(self):
        """Gracefully shuts down all application components."""
        if not self._running.is_set():
            self.logger.info("Stop command received, but application is already stopping or stopped.")
            return

        self.logger.info("Initiating application shutdown sequence...")
        self._running.clear()

        if self.detection_controller:
            self.logger.info("Stopping Detection Controller...")
            try:
                self.detection_controller.stop_detection()
            except Exception as e:
                self.logger.error(f"Error stopping Detection Controller: {e}", exc_info=True)

        if self.capture_controller:
            self.logger.info("Stopping Capture Controller...")
            try:
                self.capture_controller.stop()
            except Exception as e:
                self.logger.error(f"Error stopping Capture Controller: {e}", exc_info=True)

        if self._preview_enabled and OPENCV_AVAILABLE:
            self.logger.info("Closing OpenCV preview window...")
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                self.logger.error(f"Error closing OpenCV windows: {e}", exc_info=True)

        self.logger.info("Cat Detector application has shut down gracefully.")


if __name__ == "__main__":
    app = Application()
    try:
        app.run()
    except SystemExit:
        pass
    except Exception as e:
        print(f"[CRITICAL - Top Level] An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)