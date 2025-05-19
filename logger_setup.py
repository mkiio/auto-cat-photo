# logger_setup.py
import logging
import sys
from pathlib import Path
from config import Config # Assuming config.py is in the same directory or accessible

def setup_logging(config: Config):
    """
    Configures centralized logging for the application.
    """
    log_level_str = config.get('logging.log_level', "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = config.get('logging.log_format', "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create a root logger
    # Using a specific logger name for the application is often better than configuring the root logger directly,
    # especially if this code might be used as a library.
    # However, for a standalone app, configuring the root logger can be simpler.
    # Let's use a specific logger for the app.
    logger = logging.getLogger("CatDetectorApp")
    logger.setLevel(log_level)

    # Prevent multiple handlers if setup_logging is called more than once
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(log_format)

    # Console Handler
    if config.get('logging.log_to_console', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File Handler
    if config.get('logging.log_to_file', True):
        log_file_name = config.get('logging.log_file', "cat_detector.log")
        log_file_path = Path(log_file_name)
        if not log_file_path.is_absolute():
            # Place log file in the same directory as the config or a dedicated logs dir
            log_file_path = config.base_path / log_file_name

        # Ensure log directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path, mode='a') # 'a' for append
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # If no handlers are configured (e.g., both console and file are false),
    # add a NullHandler to prevent "No handlers could be found" warnings.
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    # Initialize a basic logger for early messages before full setup
    # This is already handled by getLogger directly.
    # The first call to getLogger will set up the default if not configured.
    # But now we are explicitly configuring it.

    # Log initial message
    logger.info(f"Logging initialized. Level: {log_level_str}, Format: '{log_format}'")
    if config.get('logging.log_to_file', True):
        logger.info(f"Logging to file: {log_file_path.resolve()}")
    if config.get('logging.log_to_console', True):
        logger.info("Logging to console enabled.")

    return logger # Return the configured logger instance

if __name__ == '__main__':
    # Example usage:
    # Create a dummy args for Config initialization if not using CLI for this test
    class DummyArgs:
        preview = None
        save_photo = None
        save_dir = None
        conf_threshold = None
        min_interval = None

    try:
        cfg = Config(args=DummyArgs()) # Use dummy args or parse real ones
        logger = setup_logging(cfg)

        logger.debug("This is a debug message.")
        logger.info("This is an info message.")
        logger.warning("This is a warning message.")
        logger.error("This is an error message.")
        logger.critical("This is a critical message.")

        # Example from another module
        module_logger = logging.getLogger("CatDetectorApp.MyModule")
        module_logger.info("Message from MyModule.")

    except Exception as e:
        # Fallback basic print if logger setup itself fails
        print(f"Failed to setup or use logger: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()