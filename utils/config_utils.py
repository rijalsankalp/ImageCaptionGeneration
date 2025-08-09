import yaml
import os
import argparse
from yaml import safe_load


def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = safe_load(f)
    return config


def save_config(config, config_path):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the YAML configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config(base_config, override_config=None, override_args=None):
    """
    Update base configuration with overrides from another config or command line arguments
    
    Args:
        base_config: Base configuration dictionary
        override_config: Optional config dictionary to override base config
        override_args: Optional namespace with command line arguments
        
    Returns:
        Updated configuration dictionary
    """
    config = base_config.copy()
    
    # Override with another config if provided
    if override_config:
        for section, section_config in override_config.items():
            if section in config:
                config[section].update(section_config)
            else:
                config[section] = section_config
    
    # Override with command line arguments if provided
    if override_args:
        args_dict = vars(override_args)
        
        # Map from argument names to config sections and keys
        arg_to_config_map = {
            "data_path": ("data", "data_path"),
            "image_dir": ("data", "image_dir"),
            "freeze_encoder": ("model", "freeze_encoder"),
            "batch_size": ("training", "batch_size"),
            "learning_rate": ("training", "learning_rate"),
            "num_epochs": ("training", "num_epochs"),
            "output_dir": ("model", "output_dir"),
            "num_samples": ("evaluation", "num_samples"),
            "save_images": ("evaluation", "save_images")
        }
        
        for arg_name, config_path in arg_to_config_map.items():
            if arg_name in args_dict and args_dict[arg_name] is not None:
                section, key = config_path
                if section not in config:
                    config[section] = {}
                config[section][key] = args_dict[arg_name]
    
    return config


def get_config(default_config_path="config/default_config.yaml", override_config_path=None, args=None):
    """
    Get configuration from default config file and optional overrides
    
    Args:
        default_config_path: Path to the default configuration file
        override_config_path: Optional path to override configuration file
        args: Optional namespace with command line arguments
        
    Returns:
        Dictionary containing configuration parameters
    """
    base_config = load_config(default_config_path)
    
    override_config = None
    if override_config_path:
        override_config = load_config(override_config_path)
        
    return update_config(base_config, override_config, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test configuration loading")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", 
                      help="Path to configuration file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    print(yaml.dump(config, default_flow_style=False))
