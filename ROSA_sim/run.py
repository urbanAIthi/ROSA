import os
import wandb
import shutil
import configparser

from env.evaluate import evaluate
from env.eval_performance import evaluate_performance
from utils.helpers import create_experiment_name, set_all_seeds, setup_logging, Timer
from utils.validators import validate_config


def run(config: configparser.ConfigParser) -> None:
    """
    Executes the complete evaluation pipeline for the selected ROSA agent.

    Args:
        config (configparser.ConfigParser): Parsed configuration file containing all experiment settings.
    """

    # Ensure the selected agent is valid
    assert config.get('General', 'agent') in ['classic'], "Unknown ROSA agent"

    # Declare the experiment name
    experiment_name = create_experiment_name(config)

    # Set random seed for reproducibility
    set_all_seeds(config.getint('Simulation', 'seed'))

    # Check if the ROSA config is valid
    validate_config(config)

    # Store the config values for wandb
    wandb_config = {k: v for section in config._sections.values() if isinstance(section, dict) for k, v in section.items()}

    # Initialize experiment tracking with wandb
    wandb.init(
        name=experiment_name,
        project=config.get('wandb', 'project'),
        config=wandb_config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
        mode=config.get('wandb', 'mode')
    )
    wandb.login(key=config.get('wandb', 'key'))

    # Create the folder structure for the experiment
    if os.path.exists(os.path.join('runs', experiment_name)):
        shutil.rmtree(os.path.join('runs', experiment_name))
    os.makedirs(os.path.join('runs', experiment_name, 'pre_eval'))
    os.makedirs(os.path.join('runs', experiment_name, 'eval'))
    sumo_path = os.path.join('runs', experiment_name, 'sumo')
    os.makedirs(sumo_path, exist_ok=True)

    for filename in os.listdir('sumo'):
        if not filename.startswith('.'):
            src_file = os.path.join('sumo', filename)
            dst_file = os.path.join(sumo_path, filename)
            shutil.copy(src_file, dst_file)

    # Save the config used for the experiment into the run directory
    shutil.copy('config.ini', os.path.join('runs', experiment_name, 'config.ini'))

    # Setup logging
    logger = setup_logging(experiment_name)
    logger.info(f'Starting experiment {experiment_name}')

    # ===== Pre-Evaluation =====
    # Run baseline evaluation without ROSA and save the results
    timer = Timer(experiment_name)
    time_on_sites_pre, co2_emissions_pre, fuel_consumptions_pre, waiting_times_pre, energy_consumptions_pre, number_stops_pre = evaluate(
        config, logger, os.path.join('runs', experiment_name, 'pre_eval'), sumo_path, pre_eval=True)
    timer.stop()

    # ******************************************
    # Classic ROSA
    # Evaluate the classic ROSA agent (classic has no training phase)
    if config.get('General', 'agent') == 'classic':
        time_on_sites, co2_emissions, fuel_consumptions, waiting_times, energy_consumptions, number_stops = evaluate(
            config, logger, os.path.join('runs', experiment_name, 'eval'), sumo_path, pre_eval=False)

    # ===== Performance Comparison =====
    # Evaluate the difference between baseline and ROSA agent
    evaluations = eval(config.get('Simulation', 'evaluations'))
    metrics = ['energy_consumption', 'co2_emission', 'fuel_consumption', 'time_on_site', 'waiting_time', 'number_stops']
    pre_eval_data = [energy_consumptions_pre, co2_emissions_pre, fuel_consumptions_pre, time_on_sites_pre,
                     waiting_times_pre, number_stops_pre]
    eval_data = [energy_consumptions, co2_emissions, fuel_consumptions, time_on_sites, waiting_times, number_stops]

    # Write performance comparison to Excel
    evaluate_performance(experiment_name, evaluations, metrics, pre_eval_data, eval_data)

    # Save config to wandb for reproducibility and sharing
    wandb.save(os.path.join('config.ini'))


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")
    run(config)


