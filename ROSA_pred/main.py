import logging
import os
import re
import yaml
import shutil
import pandas as pd
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from generate_dataset import DatasetPreprocessor, ROSADatasetGenerator
from models import MaskedSequenceTransformer
from trainer import Trainer
from tester import Tester
from utils.setup_utils import prepare_path_structure, set_seed
from utils.criterion_utils import ClassPositionLoss, MultiFeatureLoss
from utils.dataset_utils import SequenceDataset
from utils.scheduler_utils import create_scheduler
from utils.wandb_utils import start_wandb
from utils.intraining_evaluation import InTrainingEvaluator
torch.backends.cudnn.benchmark = True


def main(train_configs: str, roundabout_configs: str) -> None:
    """
    Main pipeline for training and testing multi-agent trajectory prediction model
    on roundabout scenarios using a Transformer-based architecture.

    Args:
        train_configs (str): Path to the YAML file containing the training configuration.
        roundabout_configs (str): Path to the YAML file containing the roundabout scenario configuration.
    """

    # Set device and logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Ensure reproducibility
    set_seed(42)

    # Load configuration from YAML
    with open(train_configs, 'r') as f:
        config = yaml.safe_load(f)

    with open(roundabout_configs, 'r') as f:
        roundabout_config = yaml.safe_load(f)

    # Load roundabout configs
    roundabout_scenario = config['roundabout_scenario']
    config["roundabout"] = roundabout_config["roundabouts"][roundabout_scenario]

    network_configs = config['network_configs']
    weights = config['weights']

    # Set output directory and log file
    filename = f"{config['sequence_len']}__{config['prediction_len']}__{config['min_timesteps_seen']}__{config['weights']['position_weight']}__{config['weights']['speed_weight']}__{datetime.now().strftime('%d-%m_%H-%M-%S')}"
    path = prepare_path_structure(filename, base_path='trained_models', config_files=[train_configs, roundabout_configs])
    log_file = os.path.join(path, 'training_testing.log')

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO) 
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Start wandb
    start_wandb(config, network_configs, filename, project_name=config['wandb_project'], key=config['wandb_key'], mode=config['wandb_mode'])

    # Preprocess the openDD dataset and create dataset for ROSA
    dataset_dir = os.path.join(config['dataset_path'], config['roundabout']['dataset_name'][0])
    dataset = os.path.join(dataset_dir, "dataset.pkl")

    if not os.path.exists(dataset):
        os.makedirs(dataset_dir, exist_ok=True)
        original_base = os.path.join(roundabout_scenario, config['roundabout']['original_base'])
        rosa_base = os.path.join(roundabout_scenario, os.path.splitext(config['roundabout']['original_base'])[0] + '_with_exit.sqlite')

        logger.info('Starting generating ROSA dataset')
        logger.info(rosa_base)
        logger.info(original_base)

        preprocessor = DatasetPreprocessor(
            roundabout_scenario=roundabout_scenario,
            original_base=original_base,
            rosa_base=rosa_base,
            shapefile_path=config['roundabout']['shapefile_path'],
            centerpoint=config['roundabout']['centerpoint']
        )
        preprocessor.create_dataset_with_exit()
        #preprocessor.visualize_trajectories()

        # Create dataset for ROSA
        generator = ROSADatasetGenerator(
            rosa_base=rosa_base,
            shapefile_path=config['roundabout']['shapefile_path'],
            centerpoint=config['roundabout']['centerpoint'])

        generator.save_dataset(
            pd.concat(
                [generator.process_data(table, generator.load_table(table), full_seconds_only=True) for table in
                 generator.tables],
                ignore_index=True),
            dataset)

        logger.info('Finished generating ROSA dataset')

    else:
        logger.info('ROSA dataset already exists')


    logger.info([os.path.join(config['dataset_path'], n) for n in config['roundabout']['dataset_name']])

    # Initialize training dataset and loader
    train_dataset = SequenceDataset(
        dataset_path=[os.path.join(config['dataset_path'], n) for n in config['roundabout']['dataset_name']],
        sequence_len=config['sequence_len'],
        max_agents=config['max_agents'],
        prediction_len=config['prediction_len'],
        max_prediction=config['max_prediction'],
        min_timesteps_seen=config['min_timesteps_seen'],
        split=config['train_split'],
        radius=config['radius'],
        centerpoint=config['roundabout']['centerpoint']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=12)

    # Initialize validation dataset and loader
    val_dataset = SequenceDataset(
        dataset_path=[os.path.join(config['dataset_path'], n) for n in config['roundabout']['dataset_name']],
        sequence_len=config['sequence_len'],
        max_agents=config['max_agents'],
        prediction_len=config['prediction_len'],
        max_prediction=config['max_prediction'],
        min_timesteps_seen=config['min_timesteps_seen'],
        split=config['val_split'],
        radius=config['radius'],
        centerpoint=config['roundabout']['centerpoint']
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=12)

    # Create the model
    logger.info('Creating model')
    model = MaskedSequenceTransformer(
        sequence_len=config['sequence_len'],
        max_agents=config['max_agents'],
        **network_configs['MaskedTransformer']
    )
    model.to(device)

    # Load model weights if provided
    if config['load_complete_model']:
        model.load_state_dict(torch.load(config['load_complete_model']), strict=False)

    # Define loss function
    #criterion = ClassPositionLoss(position_weight=config['weights']['position_weight'], class_weight=config['weights']['class_weight'])
    criterion = MultiFeatureLoss(**weights)

    # Get the total count of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total trainable parameters: {total_params}')

    # Set optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['scheduler']['init_lr'], weight_decay=1e-3)
    scheduler = create_scheduler(optimizer, config)

    # Initialize evaluator
    evaluator = InTrainingEvaluator(config=config, path=path)

    # Setup training pipeline
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        evaluator=evaluator,
        config=config,
        path=path,
        logger=logger
    )

    # Train the model
    logger.info('Starting the training')
    trainer.train()
    logger.info('Finished training')

    # Find the best trained model
    experiment_dir = f'trained_models/{filename}'
    pattern = re.compile(r"model_epoch_(\d+)\.pth")
    best_model = None
    best_epoch = -1

    models_dir = os.path.join(experiment_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    for filename in os.listdir(experiment_dir):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_model = filename

            src_path = os.path.join(experiment_dir, filename)
            dst_path = os.path.join(models_dir, filename)
            shutil.move(src_path, dst_path)

    # Path to the best model
    model_path = os.path.join(models_dir, best_model)

    # Autoregressive testing for increasing prediction horizons
    for prediction_len in range(1, config['max_prediction'] + 1):
        test_path = os.path.join(os.path.dirname(model_path), f"test_visualizations_{prediction_len}")
        if config['visualization_mode']:
            os.makedirs(test_path)

        # Initialize test dataset and loader
        test_dataset = SequenceDataset(
            dataset_path=[os.path.join(config['dataset_path'], n) for n in config['roundabout']['dataset_name']],
            sequence_len=config['sequence_len'],
            max_agents=config['max_agents'],
            prediction_len=prediction_len,
            max_prediction=None,
            min_timesteps_seen=config['min_timesteps_seen'],
            split=config['test_split'],
            radius=config['radius'],
            centerpoint=config['roundabout']['centerpoint']
        )
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=12)

        # Load the trained model
        logger.info('Loading model')
        model = MaskedSequenceTransformer(
            sequence_len=config['sequence_len'],
            max_agents=config['max_agents'],
            **config['network_configs']['MaskedTransformer']
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        # Setup testing pipeline
        tester = Tester(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            evaluator=evaluator,
            roundabout_scenario=roundabout_scenario,
            path=test_path,
            directory=experiment_dir,
            prediction_len=prediction_len,
            radius=config['radius'],
            centerpoint=config['roundabout']['centerpoint'],
            shapefile_path=config['roundabout']['shapefile_path'],
            visualization_mode=config['visualization_mode'],
            logger=logger
        )

        # Test the model
        logger.info('Starting the testing')
        tester.test()
        logger.info('Finished testing')


if __name__ == '__main__':
    main('configs/train_config.yaml', 'configs/roundabout_config.yaml')
