import time
import datetime
import random
import torch
import cProfile
import pstats
import logging
import numpy as np
import os
import sys
from typing import List
from sumolib import checkBinary
from stable_baselines3.common.utils import set_random_seed

import configparser
config = configparser.ConfigParser()
config.read("config.ini")

if config.getboolean('Simulation', 'libsumo'):
    import libsumo as traci
else:
    import traci


def configure_sumo(
        gui: bool,
        model_path: str,
        sumocfg_file_name: str,
        begin_time: int,
        seed: int,
        max_steps: int = 3600
) -> List[str]:
    """
    Configures the SUMO simulation command based on user-defined parameters.

    Args:
        gui (bool): If True, runs SUMO in GUI mode; otherwise, runs in command-line mode.
        model_path (str): Path to the directory containing the SUMO configuration file.
        sumocfg_file_name (str): Name of the SUMO configuration file (*.sumocfg).
        begin_time (int): Simulation start time in seconds.
        seed (int): Random seed for simulation reproducibility.
        max_steps (int, optional): Maximum number of simulation steps. Defaults to 3600.

    Returns:
        List[str]: A list of command-line arguments used to launch SUMO with the specified settings.
    """

    # Select the appropriate SUMO binary (GUI or command-line)
    if gui:
        sumo_binary = checkBinary('sumo-gui')
    else:
        sumo_binary = checkBinary('sumo')

    # Construct the full path to the configuration file
    model_path = os.path.join(model_path, sumocfg_file_name)
    # Build the command to run the SUMO simulation
    sumo_cmd = [
        sumo_binary, "-c", model_path, "--no-step-log", "true",
        "--waiting-time-memory", str(max_steps), "--xml-validation", "never", "--start", "--quit-on-end", "--begin", str(begin_time), "-W", "--seed", str(seed)]

    return sumo_cmd

def ensure_traci_closed() -> None:
    """
       Ensures that the TraCI connection is closed.
    """

    try:
        traci.close()
    except:
        # Silently ignore any errors if traci is not connected or already closed
        pass

def import_sumo_tools() -> None:
    """
    Adds the SUMO tools directory to the Python path using the SUMO_HOME environment variable.
    This is required for importing Python modules from the SUMO toolset (e.g., `sumolib`, `traci`).
    """

    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")

def create_experiment_name(config: configparser.ConfigParser) -> str:
    """
    Generates a unique experiment name based on the configuration.
    If a specific name is provided in the WandB configuration, it is used
    as the base, and the current timestamp is appended. Otherwise, the agent name
    from the general config is used with a timestamp.

    Args:
        config (configparser.ConfigParser): Configuration parser object with simulation settings.

    Returns:
        experiment_name (str): A unique name for the experiment, useful for logging and saving results.
    """

    timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime('%m-%d-%H-%M-%S')
    if config.get('wandb', 'name') != 'None':
        experiment_name = f"{config.get('wandb', 'name')}_{timestamp}"
    else:
        agent_name = config.get('General', 'agent')
        experiment_name = f"{agent_name}_{timestamp}"

    return experiment_name

def set_all_seeds(seed):
    """
    Sets all random seeds to ensure reproducibility across Python, NumPy, PyTorch, and Stable Baselines 3.

    Args:
        seed (int): The seed value to be set.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_random_seed(seed, using_cuda=torch.cuda.is_available())

def setup_logging(experiment_name) -> logging.Logger:
    """
    Sets up logging for the experiment.
    Creates a logger that outputs INFO-level messages to both the console and a log file
    located in the corresponding experiment folder.

    Args:
        experiment_name (str): The name of the experiment used to create a dedicated log directory.

    Returns:
        logging.Logger: Configured logger instance.
    """

    # Configure root logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # Create file handler for writing logs to file
    log_file = os.path.join('runs', experiment_name, 'log.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    return logger


class Timer:
    """
    A lightweight utility for profiling code execution time using cProfile.
    Enables profiling when initialized and saves profiling statistics to file upon stopping.

    Args:
        experiment_name (str): Name of the experiment, used to create the profile output file path.
    """

    def __init__(self, experiment_name) -> None:
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.path = os.path.join('runs', experiment_name, 'profile_stats.prof')

    def stop(self) -> None:
        """
        Stops the profiler and dumps collected statistics to a `.prof` file.
        """
        self.profiler.disable()
        stats = pstats.Stats(self.profiler)
        stats.dump_stats(self.path)


if __name__ == "__main__":
    raise NotImplementedError("This script is not intended to be run directly.")




