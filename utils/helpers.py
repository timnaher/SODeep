import yaml

def load_config(config_path):
    """
    Load a YAML configuration file.
    Args:
        config_path (str): The path to the YAML configuration file.
    Returns:
        dict: The configuration data loaded from the file.
    """

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config