# config.py
import yaml
from pathlib import Path
import argparse
import os

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    """
    Manages application configuration.
    Loads settings from a YAML file, allows overrides from environment variables,
    and finally from command-line arguments.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = "settings.yaml", args: argparse.Namespace = None):
        if hasattr(self, '_initialized') and self._initialized:
            return # Already initialized
        self._initialized = True

        self.base_path = Path(__file__).resolve().parent
        self.config_path = self.base_path / config_path

        self._settings = self._load_from_yaml()
        self._override_from_env()
        if args: # Command-line arguments override YAML and ENV
            self._override_from_args(args)

        self._validate_config()
        self.SAVE_DIR_ABSOLUTE = self._resolve_save_dir()


    def _load_from_yaml(self) -> dict:
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML configuration: {e}")

    def _override_from_env(self):
        """Overrides YAML settings with environment variables if they exist."""
        # Example: CATDET_DETECTION_CONFIDENCE_THRESHOLD=0.6
        for section, settings in self._settings.items():
            for key, value in settings.items():
                env_var_name = f"CATDET_{section.upper()}_{key.upper()}"
                env_value = os.getenv(env_var_name)
                if env_value is not None:
                    # Attempt to cast to original type
                    original_type = type(value)
                    try:
                        if original_type == bool:
                            self._settings[section][key] = env_value.lower() in ['true', '1', 'yes']
                        elif original_type == int:
                            self._settings[section][key] = int(env_value)
                        elif original_type == float:
                            self._settings[section][key] = float(env_value)
                        elif original_type == list: # Assuming list of simple types for env
                             self._settings[section][key] = [type(value[0])(v.strip()) for v in env_value.split(',')]
                        else:
                            self._settings[section][key] = env_value
                        print(f"[Config] Overriding '{section}.{key}' with ENV var '{env_var_name}': {self._settings[section][key]}")
                    except ValueError:
                        print(f"[Config][Warning] Could not cast ENV var {env_var_name} ('{env_value}') to type {original_type}. Using YAML value.")


    def _override_from_args(self, args: argparse.Namespace):
        """Overrides settings with command-line arguments if provided."""
        if hasattr(args, 'preview') and args.preview is not None:
            self._settings['application']['enable_preview'] = args.preview
            print(f"[Config] Overriding 'application.enable_preview' with CLI arg: {args.preview}")
        if hasattr(args, 'save_photo') and args.save_photo is not None:
            self._settings['application']['save_photos'] = args.save_photo
            print(f"[Config] Overriding 'application.save_photos' with CLI arg: {args.save_photo}")
        if hasattr(args, 'save_dir') and args.save_dir is not None:
            self._settings['application']['save_dir'] = args.save_dir
            print(f"[Config] Overriding 'application.save_dir' with CLI arg: {args.save_dir}")
        if hasattr(args, 'conf_threshold') and args.conf_threshold is not None:
            self._settings['detection']['confidence_threshold'] = args.conf_threshold
            print(f"[Config] Overriding 'detection.confidence_threshold' with CLI arg: {args.conf_threshold}")
        if hasattr(args, 'min_interval') and args.min_interval is not None:
             self._settings['detection']['min_interval_sec'] = args.min_interval
             print(f"[Config] Overriding 'detection.min_interval_sec' with CLI arg: {args.min_interval}")


    def _resolve_save_dir(self) -> Path:
        save_dir_str = self.get('application.save_dir', "photos")
        path = Path(save_dir_str)
        if not path.is_absolute():
            path = self.base_path / path
        return path

    def _validate_config(self):
        """Basic validation of critical settings."""
        if not Path(self.get('detection.model_blob_path')).exists():
             raise ConfigError(f"Model blob file not found: {self.get('detection.model_blob_path')}")
        # Add more validations as needed (e.g., camera numbers, resolutions)

    def get(self, key_path: str, default=None):
        """
        Access configuration values using a dot-separated key path.
        e.g., config.get('ai_camera.resolution')
        """
        keys = key_path.split('.')
        value = self._settings
        try:
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            if default is not None:
                return default
            raise ConfigError(f"Configuration key not found: {key_path}")
        except TypeError: # Handle case where a section is None
             if default is not None:
                return default
             raise ConfigError(f"Configuration path intermediate key is not a dictionary: {key_path}")


# --- Command-line argument parsing (can be moved to app.py or a cli_parser.py) ---
def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular Cat Detector with AI and HQ Camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows defaults in help
    )
    # Load defaults from a temporary config instance to show them in help correctly
    # This is a bit of a chicken-and-egg, but works for help text.
    # A more robust solution might involve defining defaults separately.
    try:
        temp_config_for_defaults = Config(config_path="settings.yaml") # Load defaults from YAML
        default_preview = temp_config_for_defaults.get('application.enable_preview', True)
        default_save_photo = temp_config_for_defaults.get('application.save_photos', True)
        default_save_dir = temp_config_for_defaults.get('application.save_dir', "photos")
        default_conf_threshold = temp_config_for_defaults.get('detection.confidence_threshold', 0.5)
        default_min_interval = temp_config_for_defaults.get('detection.min_interval_sec', 10)
    except ConfigError: # Fallback if settings.yaml is missing during arg parsing setup
        default_preview = True
        default_save_photo = True
        default_save_dir = "photos"
        default_conf_threshold = 0.5
        default_min_interval = 10


    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument(
        "--preview", dest="preview", action="store_true", default=None, # Default None to detect if set
        help=f"Enable live preview from AI camera (overrides YAML: currently {'ON' if default_preview else 'OFF'})."
    )
    preview_group.add_argument(
        "--no-preview", dest="preview", action="store_false",
        help="Disable live preview (overrides YAML)."
    )

    save_photo_group = parser.add_mutually_exclusive_group()
    save_photo_group.add_argument(
        "--save-photo", dest="save_photo", action="store_true", default=None,
        help=f"Enable HQ photo capture (overrides YAML: currently {'ON' if default_save_photo else 'OFF'})."
    )
    save_photo_group.add_argument(
        "--no-save-photo", dest="save_photo", action="store_false",
        help="Disable HQ photo capture (overrides YAML)."
    )

    parser.add_argument(
        "--save-dir", type=str, default=None,
        help=f"Directory to save photos (overrides YAML: currently '{default_save_dir}')."
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=None,
        help=f"Detection confidence threshold (overrides YAML: currently {default_conf_threshold})."
    )
    parser.add_argument(
        "--min-interval", type=int, default=None,
        help=f"Minimum seconds between HQ captures (overrides YAML: currently {default_min_interval})."
    )
    # Set defaults explicitly to None, so we know if they were passed or not for overriding
    parser.set_defaults(preview=None, save_photo=None)

    return parser.parse_args()

if __name__ == '__main__':
    # Example usage:
    args_for_config = parse_cli_args() # Parse CLI args first
    try:
        config = Config(args=args_for_config) # Then initialize config with them
        print("--- Configuration Loaded ---")
        print(f"AI Camera Number: {config.get('ai_camera.camera_num')}")
        print(f"HQ Camera Number: {config.get('hq_camera.camera_num')}")
        print(f"Enable Preview (final): {config.get('application.enable_preview')}")
        print(f"Save Photos (final): {config.get('application.save_photos')}")
        print(f"Save Directory (absolute): {config.SAVE_DIR_ABSOLUTE}")
        print(f"Confidence Threshold (final): {config.get('detection.confidence_threshold')}")
        print(f"Min Interval (final): {config.get('detection.min_interval_sec')}")
        print(f"Model Path: {config.get('detection.model_blob_path')}")
        print(f"Log Level: {config.get('logging.log_level')}")

        # Example of setting an ENV VAR before running this test script:
        # export CATDET_DETECTION_CONFIDENCE_THRESHOLD=0.75
        # python config.py --no-preview --save-dir /tmp/mycats

    except ConfigError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")