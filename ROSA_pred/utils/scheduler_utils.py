import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math
from typing import Union, Dict, List


def create_scheduler(optimizer: optim.Optimizer, config: Dict) -> Union[lr_scheduler._LRScheduler, lr_scheduler.ReduceLROnPlateau]:
    """
    Create a learning rate scheduler based on the configuration dictionary.

    Args:
        optimizer (optim.Optimizer): The optimizer for which to schedule the learning rate.
        config (Dict): Configuration dictionary specifying the scheduler type and parameters.

    Returns:
        lr_scheduler: The initialized scheduler object.

    Raises:
        NotImplementedError: If the scheduler type specified in config is not implemented.
    """

    if config['scheduler']['type'] == 'default':
        return lr_scheduler.StepLR(optimizer, step_size=config['default_scheduler_config']['step_size'], gamma=config['default_scheduler_config']['gamma'])
    elif config['scheduler']['type'] == 'cosine':
        return WarmupCosineAnnealingLR(config)
    elif config['scheduler']['type'] == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['plateau_scheduler_config']['factor'], patience=config['plateau_scheduler_config']['patience'], min_lr=float(config['plateau_scheduler_config']['min_lr']))
    else:
        raise NotImplementedError("Scheduler type not implemented")


class WarmupCosineAnnealingLR(lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with warmup and cosine annealing.

    Args:
        optimizer (optim.Optimizer): Wrapped optimizer.
        config (Dict): Configuration dictionary with keys:
            - 'startup_steps' (int): Number of warmup steps.
            - 'min_lr' (float): Minimum learning rate after annealing.
            - 'num_epochs' (int): Total number of epochs (steps).
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer: optim.Optimizer, config: Dict[str, float | int], last_epoch: int = -1) -> None:
        self.startup_steps = config.get('startup_steps')
        self.min_lr = config.get('min_lr')
        self.total_steps = config.get('num_epochs')  # Total training steps
        self.cosine_annealing_steps = self.total_steps - self.startup_steps
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate for current step.

        Returns:
            list[float]: List of learning rates for each parameter group at the current step.
        """

        if self.last_epoch < self.startup_steps: #Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.startup_steps for base_lr in self.base_lrs]
        else: # Cosine annealing after warmup
            return [self.min_lr + (base_lr - self.min_lr) * (1 + math.cos(math.pi * (self.last_epoch - self.startup_steps) / self.cosine_annealing_steps)) / 2 for base_lr in self.base_lrs]
