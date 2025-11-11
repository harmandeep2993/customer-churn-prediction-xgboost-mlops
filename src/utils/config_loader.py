import yaml

def load_config(config_path: str):
    """Load and return YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)