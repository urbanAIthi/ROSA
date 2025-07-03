import os
import shutil
import numpy as np
import torch
import random
from typing import List


def prepare_path_structure(filename: str, base_path: str, config_files: List[str]) -> str:
    """
    Creates a directory structure and copies configuration files into it.

    Args:
        filename (str): Name of the subdirectory to create inside base_path.
        base_path (str): Base directory where the new subdirectory will be created.
        config_files (List[str]): List of config file paths to copy into the new directory.

    Returns:
        path (str): The full path to the newly created directory.
    """

    os.makedirs(base_path, exist_ok=True)
    path = os.path.join(base_path, filename)
    os.mkdir(path)

    # Copy the config files to the path
    for file in config_files:
        shutil.copy(file, os.path.join(path, os.path.basename(file)))

    return path

def set_seed(seed_value: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed_value (int, optional): The seed value to use. Defaults to 42.
    """

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

