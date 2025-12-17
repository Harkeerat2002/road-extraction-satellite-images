import json
from pathlib import Path


# Per-user config file stored alongside this module.
CONFIG_FILE = Path(__file__).with_name(".paths_config.json")


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    """Simple interactive yes/no question."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    answer = input(prompt + suffix).strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def _create_config_interactively() -> dict:
    """
    Ask the user where the data lives and whether they are on Colab,
    then write a small JSON config file.
    """
    print("Configuring dataset paths for this project.")

    is_colab = _ask_yes_no("Are you running on Google Colab?", default=False)

    if is_colab:
        data_root = "/content/drive/MyDrive/CS433"
        log_root = "/content/logs"
        print(f"Using Colab defaults:\n  data_root = {data_root}\n  log_root  = {log_root}")
    else:
        print(
            "Please enter the root path that contains the CS433 data\n"
            "(it should have subfolders like 'training', 'test_set_images', etc.)."
        )
        data_root = input("Data root path: ").strip()
        while not data_root:
            print("Data root cannot be empty.")
            data_root = input("Data root path: ").strip()

        # For local setups, default logs to a 'logs' subfolder under data_root.
        default_log_root = str(Path(data_root) / "logs")
        log_root = input(f"Logging directory [{default_log_root}]: ").strip() or default_log_root

    cfg = {"data_root": data_root, "log_root": log_root}
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    print(f"Saved path configuration to {CONFIG_FILE}")
    return cfg


def _load_raw_config() -> dict:
    """
    Load the JSON config if it exists; otherwise create it interactively.
    """
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except json.JSONDecodeError:
            print(f"Warning: {CONFIG_FILE} is corrupted or invalid. Recreating it.")
            return _create_config_interactively()
    else:
        return _create_config_interactively()


def _derive_paths() -> dict:
    """
    Derive all concrete paths needed by the rest of the codebase
    from the raw configuration.
    """
    raw_cfg = _load_raw_config()

    data_root = Path(raw_cfg["data_root"])
    log_root = Path(raw_cfg.get("log_root", data_root / "logs"))

    paths = {
        "DATA_ROOT": data_root,
        "TRAIN_DIR": data_root / "training",
        "TEST_DIR": data_root / "test_set_images",
        "OUTPUT_DIR": data_root / "results",
        "CHECKPOINT_DIR": data_root / "checkpoints",
        "LOGGING_DIR": log_root,
    }

    return {k: str(v) for k, v in paths.items()}


# Public constants used by configs.py
_PATHS = _derive_paths()

DATA_ROOT = _PATHS["DATA_ROOT"]
TRAIN_DIR = _PATHS["TRAIN_DIR"]
TEST_DIR = _PATHS["TEST_DIR"]
OUTPUT_DIR = _PATHS["OUTPUT_DIR"]
CHECKPOINT_DIR = _PATHS["CHECKPOINT_DIR"]
LOGGING_DIR = _PATHS["LOGGING_DIR"]

