# app.py
import time
import signal
import sys
import logging
import threading
from typing import Optional, Tuple, Any, Callable
import numpy as np

from config import Config, ConfigError, parse_cli_args
from logger_setup import setup_logging
from detection_controller import DetectionController, CatDetectedEventData
from capture_controller import CaptureController

# Conditional import for UI/Preview part
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

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

        self._running = threading.Event() # Event to signal running state for threads/loops
        self._preview_enabled = False
        self._flash_frames_preview = 0 # For visual feedback on detection in preview
        self._last_box_preview_cv: Optional[Tuple[int,int,int,int]] = None


    def _setup_signal_handlers(self):
        """Sets up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        self.logger.info("Signal handlers for SIGINT and SIGTERM registered.")

    def _handle_signal(self, signum, frame):
        self.logger.warning(f"Signal {signal.Signals(signum).name} received. Initiating graceful shutdown...")
        self.stop()

    def _initialize_modules(self):
        """Initializes all application components."""
        try:
            # 1. Parse CLI arguments
            cli_args = parse_cli_args()

            # 2. Load Configuration (CLI args will override YAML/ENV)
            self.config = Config(args=cli_args) # Config is a singleton, safe to call again if needed

            # 3. Setup Logging (using loaded configuration)
            self.logger = setup_logging(self.config)
            # All subsequent logging should use this logger instance or logging.getLogger("CatDetectorApp")

        except ConfigError as e:
            # If config fails, logger might not be set. Fallback to print.
            print(f"[CRITICAL] Configuration error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"[CRITICAL] Failed during initial setup (config/logging): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

        self.logger.info("Application configuration and logging initialized.")

        try:
            # 4. Initialize Controllers
            self.detection_controller = DetectionController(self.config)
            self.capture_controller = CaptureController(self.config)

            # 5. Setup Inter-Module Communication (Callbacks/Events)
            # DetectionController -> CaptureController
            self.detection_controller.subscribe_to_cat_detections(
                self.capture_controller.handle_cat_detected
            )
            self.logger.info("Subscribed CaptureController to cat detection events.")

            # 6. Preview Setup (if enabled)
            self._preview_enabled = self.config.get('application.enable_preview')
            if self._preview_enabled:
                if not OPENCV_AVAILABLE:
                    self.logger.warning("OpenCV is not available, cannot enable live preview window.")
                    self._preview_enabled = False # Force disable
                else:
                    self.logger.info("Live preview enabled. Setting up display callback.")
                    self.detection_controller.subscribe_to_frame_ready(self._display_preview_frame)
                    cv2.namedWindow("Cat Detector - AI Feed (cam0)", cv2.WINDOW_AUTOSIZE)

        except Exception as e:
            self.logger.critical(f"Failed to initialize application controllers or setup communication: {e}", exc_info=True)
            sys.exit(1)

        self.logger.info("All application modules initialized successfully.")


    def _display_preview_frame(self, frame: Optional[np.ndarray], detection_box: Optional[Tuple[int, int, int, int]]):
        """Callback for DetectionController to display frames via OpenCV."""
        if not self._preview_enabled or frame is None:
            return

        try:
            # BGR for OpenCV display
            display_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame

            if detection_box: # A new detection box is available from this frame's metadata
                self._last_box_preview_cv = detection_box
                self._flash_frames_preview = 5 # Number of frames to show the box
                self.logger.debug(f"Preview: New detection box received: {detection_box}")


            if self._flash_frames_preview > 0 and self._last_box_preview_cv:
                x0, y0, x1, y1 = self._last_box_preview_cv
                cv2.rectangle(display_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2) # Green box
                cv2.putText(
                    display_bgr, "Cat!", (x0, max(15, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                self._flash_frames_preview -= 1
            # else: # If no active flash, ensure _last_box_preview_cv is cleared if needed
            #    _last_box_preview_cv = None # This would prevent box from lingering if detection stops

            cv2.imshow("Cat Detector - AI Feed (cam0)", display_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.logger.info("Preview window: 'q' pressed. Initiating shutdown.")
                self.stop() # Graceful shutdown
            elif key == ord('s'): # Manual save trigger for debugging
                self.logger.info("Preview window: 's' pressed. Manually triggering cat detected event for HQ capture.")
                if self.capture_controller and self.detection_controller:
                    # Simulate a detection event data
                    dummy_event_data: CatDetectedEventData = (None, (10,10,100,100), 0.99, "manual_cat_trigger") # type: ignore
                    self.capture_controller.handle_cat_detected(dummy_event_data)


        except Exception as e:
            self.logger.error(f"Error in preview display callback: {e}", exc_info=True)
            # If preview fails catastrophically, maybe disable it
            # self._preview_enabled = False
            # cv2.destroyAllWindows()


    def run(self):
        """Main application entry point."""
        self._initialize_modules() # This will exit on critical errors
        self._setup_signal_handlers()
        self._running.set() # Signal that the application is now running

        self.logger.info("Starting application components...")
        try:
            if self.capture_controller:
                self.capture_controller.start() # Initializes HQ camera if enabled

            if self.detection_controller:
                self.detection_controller.start_detection() # Starts AI camera and NN processing

            self.logger.info("Application is now running. Press Ctrl+C to exit.")
            if self._preview_enabled:
                self.logger.info("Live preview is ON. Press 'q' in the preview window to quit.")
            else:
                self.logger.info("Live preview is OFF.")


            # The main application loop.
            # If preview is enabled, DetectionController's run_preview_loop_if_needed
            # will fetch frames and call _display_preview_frame, which has waitKey.
            # If preview is not enabled, we need a way to keep the main thread alive
            # and responsive to signals, while other threads (detection, capture) do their work.
            if self.detection_controller and self._preview_enabled:
                # This loop is now managed by the DetectionController's own loop
                # that calls the _display_preview_frame where cv2.waitKey() is.
                # The DetectionController's preview loop will block here.
                self.detection_controller.run_preview_loop_if_needed()
            else:
                # If no preview, just wait for the _running event to be cleared by stop()
                while self._running.is_set():
                    time.sleep(0.5) # Keep main thread alive, check signals

        except RuntimeError as e: # Catch runtime errors from controller starts
            self.logger.critical(f"Runtime error during application execution: {e}", exc_info=True)
        except KeyboardInterrupt: # Should be caught by signal handler, but as a fallback
            self.logger.info("KeyboardInterrupt in main run loop. Shutting down.")
        except Exception as e:
            self.logger.critical(f"Unhandled exception in main run loop: {e}", exc_info=True)
        finally:
            if self._running.is_set(): # If stop() wasn't called by other means (e.g. 'q' in preview)
                self.logger.info("Main loop ended, ensuring graceful shutdown.")
                self.stop() # Ensure cleanup is called
            else:
                self.logger.info("Application has already been signaled to stop.")


    def stop(self):
        """Gracefully shuts down all application components."""
        if not self._running.is_set(): # Check if already stopping/stopped
            self.logger.info("Stop command received, but application is already stopping or stopped.")
            return

        self.logger.info("Initiating application shutdown sequence...")
        self._running.clear() # Signal all loops and threads to stop

        # Stop controllers (order might matter, e.g., stop detection before capture or vice-versa)
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
        # Ensure all non-daemon threads have joined if any were started by app.py directly.
        # For daemon threads, they will exit when the main thread exits.


if __name__ == "__main__":
    # Ensure the current directory is in PYTHONPATH if modules are in the same dir
    # sys.path.insert(0, Path(__file__).resolve().parent)
    app = Application()
    try:
        app.run()
    except SystemExit: # Allow sys.exit() for early exits (e.g. config errors)
        pass # Logger (if initialized) or print would have handled the message
    except Exception as e: # Catch-all for any unexpected errors during app.run() before logger is set
        print(f"[CRITICAL - Top Level] An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)