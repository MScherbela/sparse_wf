from sparse_wf.train_with_config import train_with_config
from sparse_wf.setup_calculations import DEFAULT_CONFIG_PATH, update_dict, load_yaml, save_yaml

if __name__ == "__main__":
    default_config = load_yaml(DEFAULT_CONFIG_PATH)
    file_config = load_yaml("debug_config.yaml")
    full_config = update_dict(default_config, file_config, allow_new_keys=False)
    save_yaml("full_config.yaml", full_config)
    train_with_config("full_config.yaml")
