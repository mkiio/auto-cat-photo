# detection_controller.py
import time
import logging
from typing import Callable, Any, Tuple, Optional
import numpy as np

# Conditional import for Picamera2 and IMX500 to allow for testing on non-Pi environments
try:
    from picamera2 import Picamera2, Preview
    from picamera2.devices import IMX500
    from picamera2.controls import Controls
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    # Mock classes for testing or environments without Picamera2
    class Picamera2Mock:
        def __init__(self, camera_num=0): self.camera_num = camera_num; self.is_running = False; self.preview_configuration = None; self.still_configuration = None; self.post_callback = None; print(f"MockPicamera2(cam={camera_num}) created.")
        def create_preview_configuration(self, main=None, lores=None, raw=None, transform=None, colour_space=None, buffer_count=4, controls=None, display=None, encode=None): self.preview_configuration = {"main": main}; return self.preview_configuration
        def configure(self, cfg): print(f"MockPicamera2 configured with: {cfg}"); self._config = cfg
        def start_preview(self, preview_type=None): print(f"MockPicamera2 preview started ({preview_type})."); self.is_running = True
        def stop_preview(self): print("MockPicamera2 preview stopped."); self.is_running = False
        def start(self): print("MockPicamera2 camera started."); self.is_running = True
        def stop(self): print("MockPicamera2 camera stopped."); self.is_running = False
        def capture_array(self, stream_name="main"): print(f"MockPicamera2 capturing array from {stream_name}."); return np.zeros((self.preview_configuration['main']['size'][1], self.preview_configuration['main']['size'][0], 3), dtype=np.uint8)
        def close(self): print("MockPicamera2 closed.")
        @property
        def controls(self): return Controls(self)


    class IMX500Mock:
        def __init__(self, model_path=None): self.model_path = model_path; self.camera_num = 0; print(f"MockIMX500 created with model: {model_path}")
        def get_outputs(self, metadata): print("MockIMX500 get_outputs called."); return (np.array([[0.1, 0.1, 0.8, 0.8]]), np.array([0.9]), np.array([17])) # Mocked: box, score, class (cat)
        def convert_inference_coords(self, box, metadata, camera_instance): print("MockIMX500 convert_inference_coords called."); return (10, 10, 50, 50) # x, y, w, h

    Picamera2 = Picamera2Mock
    IMX500 = IMX500Mock


from config import Config

# Define an event type for better clarity
CatDetectedEventData = Tuple[np.ndarray, Tuple[int, int, int, int], float, str] # frame, box, score, class_name

class DetectionController:
    """
    Manages the AI Camera (Cam0) for object detection (specifically cats).
    Encapsulates AI model loading, performs inference, and emits detection events.
    """
    def __init__(self, config: Config):
        self.logger = logging.getLogger("CatDetectorApp.DetectionController")
        self.config = config

        self.ai_camera_num = self.config.get('ai_camera.camera_num')
        self.model_blob_path = self.config.get('detection.model_blob_path')
        self.preview_res = tuple(self.config.get('ai_camera.preview_resolution'))
        self.preview_format = self.config.get('ai_camera.format')
        self.conf_threshold = self.config.get('detection.confidence_threshold')
        self.coco_cat_id = self.config.get('detection.coco_cat_id')
        self.margin_ratio = self.config.get('detection.preview_box_margin_ratio')

        self.imx: Optional[IMX500] = None
        self.cam_ai: Optional[Picamera2] = None
        self.preview_cfg: Optional[dict] = None

        self.cat_detected_callbacks: list[Callable[[CatDetectedEventData], None]] = []
        self.frame_ready_callbacks: list[Callable[[np.ndarray, Optional[Tuple[int, int, int, int]]], None]] = [] # For preview

        self.is_running = False
        self._last_detection_box_preview: Optional[Tuple[int, int, int, int]] = None


    def _initialize_ai_camera(self):
        if not PICAMERA2_AVAILABLE:
            self.logger.warning("Picamera2 library not found. Using MOCK cameras. Detection will be simulated.")
        try:
            self.logger.info(f"Initializing AI Model (IMX500) with blob: {self.model_blob_path}")
            self.imx = IMX500(self.model_blob_path) # type: ignore
            # The IMX500 wrapper should ideally allow specifying camera number or derive it.
            # Assuming the Picamera2 instance for IMX500 should use the specific ai_camera_num.
            # The original script used self.imx.camera_num, implying IMX500 might manage this.
            # For clarity, we explicitly pass the configured camera number if the API supports it.
            # If IMX500 determines its own camera, this might need adjustment.
            # If IMX500 class automatically uses cam 0, and ai_camera_num is also 0, it's fine.
            # If they differ, we need to ensure Picamera2(self.ai_camera_num) is the one IMX500 expects.

            self.logger.info(f"Initializing AI Camera (cam{self.ai_camera_num}) for detection...")
            self.cam_ai = Picamera2(self.ai_camera_num) # type: ignore

            self.preview_cfg = self.cam_ai.create_preview_configuration(
                main={"format": self.preview_format, "size": self.preview_res}
            )
            self.cam_ai.configure(self.preview_cfg)
            self.cam_ai.post_callback = self._on_nn_request # Attach NN processing callback
            self.logger.info(f"AI Camera (cam{self.ai_camera_num}) initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize AI camera or IMX500: {e}", exc_info=True)
            raise RuntimeError(f"AI Camera initialization failed: {e}")


    def _on_nn_request(self, request):
        """Callback executed by Picamera2 after a frame is processed by IMX500."""
        current_detection_box_for_frame = None
        try:
            # Retrieve the original frame that was processed for detection.
            # This might require accessing the completed request or a specific buffer.
            # The original script captured a new array in the main loop for display.
            # For efficiency, we should use the frame associated with this detection.
            # Assuming request.make_array("main") gives the frame for this metadata.
            # If not, this part needs adjustment based on Picamera2 API for metadata-linked frames.
            # For now, we will capture a fresh one in run_detection_loop if needed for frame_ready_callbacks
            # and focus on detection data here.

            metadata = request.get_metadata()
            outputs = self.imx.get_outputs(metadata) # type: ignore
        except Exception as e:
            self.logger.warning(f"Failed to get NN outputs from request: {e}")
            return

        if not outputs or len(outputs) < 3:
            self.logger.debug("NN outputs were empty or incomplete.")
            self._last_detection_box_preview = None # Clear last box if no valid output
            return # No valid output to process

        boxes, scores, classes = outputs[:3] # boxes are relative to NN input

        for i, score in enumerate(scores):
            if score >= self.conf_threshold and int(classes[i]) == self.coco_cat_id:
                self.logger.info(f"Cat detected! Score: {score:.2f}, Class ID: {int(classes[i])}")

                # Convert NN coordinates to preview frame coordinates
                rel_box_nn = tuple(boxes[i]) # Box relative to what the NN processed
                try:
                    # This conversion needs the metadata from the *request* tied to *this* inference
                    x, y, w, h = self.imx.convert_inference_coords( # type: ignore
                        rel_box_nn, metadata, self.cam_ai
                    )
                    x1, y1 = x + w, y + h

                    # Get preview dimensions for margin calculation
                    w_pre, h_pre = self.preview_cfg["main"]["size"] # type: ignore
                    x0_margin, y0_margin, x1_margin, y1_margin = self._add_margin(
                        x, y, x1, y1, w_pre, h_pre, self.margin_ratio
                    )
                    detection_box_preview = (x0_margin, y0_margin, x1_margin, y1_margin)
                    self._last_detection_box_preview = detection_box_preview
                    current_detection_box_for_frame = detection_box_preview


                    # Emit cat_detected event
                    # For the event, we might want to pass the frame that led to detection.
                    # This is tricky with post_callback as the frame might not be easily accessible
                    # or could be a different buffer.
                    # For now, we'll pass None for the frame in the event,
                    # CaptureController can decide to capture a fresh one.
                    # A more advanced setup might use Queues to pass (frame, detection_data).
                    event_data: CatDetectedEventData = (None, detection_box_preview, float(score), "cat") # type: ignore # Frame is None
                    for callback in self.cat_detected_callbacks:
                        try:
                            callback(event_data)
                        except Exception as e_cb:
                            self.logger.error(f"Error in cat_detected_callback: {e_cb}", exc_info=True)
                    break # Process first detected cat only
                except Exception as e_coord:
                    self.logger.error(f"Error converting NN coords or processing detection: {e_coord}", exc_info=True)
                    self._last_detection_box_preview = None
                    break
            else: # No cat detected above threshold in this iteration
                 self._last_detection_box_preview = None


    def _add_margin(self, x0, y0, x1, y1, frame_w, frame_h, ratio):
        dx = int((x1 - x0) * ratio * 0.5)
        dy = int((y1 - y0) * ratio * 0.5)
        return (
            max(0, x0 - dx), max(0, y0 - dy),
            min(frame_w, x1 + dx), min(frame_h, y1 + dy),
        )

    def subscribe_to_cat_detections(self, callback: Callable[[CatDetectedEventData], None]):
        """Allows other modules to subscribe to 'cat_detected' events."""
        self.logger.info(f"Registering callback for cat detections: {callback.__name__}")
        self.cat_detected_callbacks.append(callback)

    def subscribe_to_frame_ready(self, callback: Callable[[np.ndarray, Optional[Tuple[int,int,int,int]]], None]):
        """Allows other modules (e.g., UI) to get frames for display."""
        self.logger.info(f"Registering callback for frame ready: {callback.__name__}")
        self.frame_ready_callbacks.append(callback)


    def start_detection(self):
        if self.is_running:
            self.logger.info("Detection is already running.")
            return
        try:
            self._initialize_ai_camera() # Initialize components
            if self.cam_ai:
                self.cam_ai.start() # Start camera, which triggers post_callback
                self.is_running = True
                self.logger.info("AI Camera started and detection loop running via post_callback.")
            else:
                self.logger.error("AI Camera not initialized, cannot start detection.")
                raise RuntimeError("AI Camera not available for starting detection.")

        except Exception as e:
            self.logger.critical(f"Failed to start detection controller: {e}", exc_info=True)
            self.is_running = False # Ensure state is correct
            # Re-raise or handle appropriately, maybe app should try to restart or exit
            raise

    def stop_detection(self):
        if not self.is_running and not self.cam_ai : #Also check if cam_ai exists
            self.logger.info("Detection is not running or camera not initialized.")
            return
        self.is_running = False # Signal to stop for any external loops if used
        try:
            if self.cam_ai:
                self.logger.info("Stopping AI Camera (cam0)...")
                self.cam_ai.stop() # Stop camera operations
                self.cam_ai.close() # Release camera resources
                self.cam_ai = None
                self.imx = None # Assuming IMX needs no explicit close beyond camera.
                self.logger.info("AI Camera (cam0) stopped and closed.")
        except Exception as e:
            self.logger.error(f"Error stopping/closing AI camera (cam0): {e}", exc_info=True)
        finally:
            self.cam_ai = None # Ensure it's None even if close failed
            self.imx = None


    def run_preview_loop_if_needed(self):
        """
        If direct preview from post_callback is not feasible for UI updates,
        this method can be run in a thread to capture frames and send them.
        However, the original script's main loop implies capture_array is done in main thread.
        The post_callback handles detections. Frames for UI are separate.
        This function will be called by app.py in the main thread if preview is enabled.
        """
        if not self.is_running or not self.cam_ai: # self.is_running is the controller's state
            self.logger.warning("Detection not running or AI camera not available for preview loop.")
            return

        self.logger.info("Starting AI camera preview feed for UI updates.")
        # self.is_running is set to False by self.stop_detection()
        # which can be called by the main app's signal handler or 'q' press
        while self.is_running:
            try:
                if not self.cam_ai: # Double check if cam_ai was set to None by stop_detection
                    self.logger.info("AI camera instance became None. Exiting preview loop.")
                    break

                # This frame is for display purposes. Detection happens in post_callback.
                frame_for_preview = self.cam_ai.capture_array("main")

                current_box_to_draw = self._last_detection_box_preview

                for callback in self.frame_ready_callbacks:
                    try:
                        callback(frame_for_preview.copy(), current_box_to_draw)
                    except Exception as e_cb:
                        self.logger.error(f"Error in frame_ready_callback: {e_cb}", exc_info=True)

                # The loop speed is primarily governed by capture_array() and waitKey() in the UI callback.
                # If waitKey is very short or non-existent (e.g. headless), ensure this loop doesn't spin too fast
                # if capture_array returns very quickly without blocking.
                # However, capture_array for a camera usually blocks until a new frame.

            except RuntimeError as e: # Picamera2 often raises RuntimeError for camera issues
                if "Camera has been stopped" in str(e) or "Camera is not streaming" in str(e):
                    self.logger.info(f"AI camera preview loop: Camera stopped or not streaming. Exiting loop. ({e})")
                else:
                    self.logger.error(f"AI camera preview loop: Runtime error capturing frame: {e}", exc_info=True)
                # If capture fails, it's a good indication we should stop trying.
                # self.stop_detection() # This would set self.is_running to False
                break # Exit the loop, app.py's main loop will then also exit due to self.is_running
            except Exception as e:
                self.logger.error(f"AI camera preview loop: Unexpected error: {e}", exc_info=True)
                # Consider breaking on other critical errors too.
                time.sleep(0.1) # Avoid busy-looping on transient errors if not breaking

        self.logger.info("AI camera preview feed for UI updates has stopped.")


if __name__ == '__main__':
    # Example Usage (requires a running X server if OpenCV is used for display by a subscriber)
    # This is a basic test, app.py will orchestrate more complex interactions.
    class DummyArgs: preview = None; save_photo = None; save_dir = None; conf_threshold = None; min_interval = None

    # Attempt to load config and setup logger
    # This assumes config.py and logger_setup.py are in the same directory or PYTHON_PATH
    try:
        from config import Config
        from logger_setup import setup_logging
        cfg = Config(args=DummyArgs())
        logger = setup_logging(cfg)
    except ImportError:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logger = logging.getLogger("DetectionControllerTest")
        logger.warning("Could not import Config or logger_setup. Using basic logging for test.")
        # Create a dummy cfg object with essential defaults if full config is not available
        class MinimalCfg:
            def get(self, key, default=None):
                settings = {
                    'application.enable_preview': True,
                    'ai_camera.camera_num': 0,
                    'detection.model_blob_path': "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk", # Example path
                    'ai_camera.preview_resolution': [640,480],
                    'ai_camera.format': "RGB888",
                    'detection.confidence_threshold': 0.5,
                    'detection.coco_cat_id': 17,
                    'detection.preview_box_margin_ratio': 0.20
                }
                parts = key.split('.')
                val = settings
                try:
                    for p in parts: val = val[p]
                    return val
                except KeyError:
                    return default
        cfg = MinimalCfg()


    if not cfg.get('application.enable_preview'):
        logger.info("Preview is disabled in config. Exiting example.")
        exit()

    try:
        controller = DetectionController(config=cfg)

        # --- Dummy Frame Display using OpenCV (simulates a UI module) ---
        if PICAMERA2_AVAILABLE: # Only setup cv2 if we might get real frames
            import cv2
            # Check if X server is available for cv2.namedWindow
            try:
                cv2.namedWindow("AI Detection Feed (cam0)", cv2.WINDOW_AUTOSIZE)
                opencv_window_created = True
            except cv2.error as e:
                logger.warning(f"Could not create OpenCV window (is X server running?): {e}. Preview will not be shown.")
                opencv_window_created = False
        else:
            opencv_window_created = False


        # These variables are for the scope of this __main__ block
        _flash_frames_preview_main = 0
        _last_box_preview_cv_main = None

        def display_frame_stub(frame: np.ndarray, detection_box: Optional[Tuple[int, int, int, int]]):
            # Use the _main suffixed variables here
            # No 'nonlocal' needed as they are in the enclosing __main__ scope
            global _flash_frames_preview_main, _last_box_preview_cv_main

            if not opencv_window_created: return # Don't try to show if window failed

            display_bgr = frame if PICAMERA2_AVAILABLE and frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if detection_box:
                _last_box_preview_cv_main = detection_box
                _flash_frames_preview_main = 5 # Show box for 5 frames

            if _flash_frames_preview_main > 0 and _last_box_preview_cv_main:
                x0, y0, x1, y1 = _last_box_preview_cv_main
                cv2.rectangle(display_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(
                    display_bgr, "Cat!", (x0, max(15, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                _flash_frames_preview_main -= 1
            else:
                _last_box_preview_cv_main = None

            cv2.imshow("AI Detection Feed (cam0)", display_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Stopping detection via 'q' key.")
                if controller.is_running: # Check if controller is actually running
                    controller.stop_detection()

        controller.subscribe_to_frame_ready(display_frame_stub)

        def on_cat_detected_stub(data: CatDetectedEventData):
            frame, box, score, class_name = data
            logger.info(f"STUB HANDLER: Cat detected event! Class: {class_name}, Score: {score:.2f}, Box: {box}")

        controller.subscribe_to_cat_detections(on_cat_detected_stub)

        logger.info("Starting detection controller...")
        controller.start_detection()

        if cfg.get('application.enable_preview'):
            logger.info("Running preview loop from main test...")
            controller.run_preview_loop_if_needed() # This blocks


    except RuntimeError as e:
        logger.critical(f"Failed to run DetectionController example: {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping detection...")
    finally:
        if 'controller' in locals() and controller.is_running:
            logger.info("Ensuring detection controller is stopped in finally block.")
            controller.stop_detection()
        if PICAMERA2_AVAILABLE and cfg.get('application.enable_preview') and opencv_window_created:
            cv2.destroyAllWindows()
            logger.info("OpenCV windows destroyed.")
        logger.info("DetectionController example finished.")