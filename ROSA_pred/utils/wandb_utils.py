import wandb
from typing import Dict


def start_wandb(cfg: Dict, network_config: Dict, filename: str, project_name: str, key: str, mode: str = 'disabled') -> None:
    """
    Initializes and starts a Weights & Biases (wandb) run with the given configuration.

    Args:
        cfg (Dict): General configuration dictionary for the experiment.
        network_config (Dict): Network-specific configuration dictionary.
        filename (str): Name used for the wandb run.
        project_name (str): Name of the wandb project.
        mode (str, optional): wandb mode, one of 'online', 'offline', or 'disabled'. Defaults to 'disabled'.
    """

    while mode not in ["online", "offline", "disabled"]:  
        mode = input('wandb mode (online, offline, disabled): ')
    print(f'wandb mode: {mode}')
    wandb.login(key=key)
    wandb.init(project=project_name, mode=mode, name=filename)
    wandb.config.update(cfg)
    wandb.config.update(network_config['MaskedTransformer'])



