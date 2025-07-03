import logging
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.criterion_utils import ClassPositionLoss, MultiFeatureLoss
from utils.intraining_evaluation import InTrainingEvaluator
torch.backends.cudnn.benchmark = True


class Trainer:
    """
    Trainer class to handle model training and validation.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_loader (DataLoader): DataLoader providing the training data.
        val_loader (DataLoader): DataLoader providing the validation data.
        criterion (MultiFeatureLoss): Loss function used for training and evaluation.
        optimizer (optim.Optimizer): Optimizer to update model weights.
        scheduler (lr_scheduler._LRScheduler): Learning rate scheduler.
        device (torch.device): Device on which to perform training (CPU or CUDA).
        evaluator (InTrainingEvaluator): Evaluator object to track and log evaluation metrics.
        config (Dict): Dictionary containing training configuration parameters.
        path (str): Directory path to save trained models.
        logger (logging.Logger): Logger for training events and status messages.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: MultiFeatureLoss,
        optimizer: optim.Optimizer,
        scheduler: lr_scheduler._LRScheduler,
        device: torch.device,
        evaluator: InTrainingEvaluator,
        config: Dict,
        path: str,
        logger: logging.Logger
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.evaluator = evaluator
        self.config = config
        self.path = path
        self.logger = logger
        self.best_val_loss = float('inf')
        self.use_scaler = False
        if self.use_scaler:
            self.scaler = torch.cuda.amp.GradScaler()

    def process_batch(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            train: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Processes a single batch of data for training or validation.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing input tensors, target tensors, and sample indices.
            train (bool): If True, performs backpropagation and optimizer step. Defaults to True.

        Returns:
            loss (torch.Tensor]: Computed loss tensor.
            additional_information (Dict): Dictionary of additional information from the loss function.
                - Loss tensor.
                - Dictionary of additional information from the loss function.
        """

        input_tensor, target_tensor, indexes = batch
        input_tensor, target_tensor = input_tensor.to(self.device), target_tensor.to(self.device)

        # Forward pass (with or without automatic mixed precision)
        if self.use_scaler:
            with torch.amp.autocast(device_type=self.device.type):
                outputs = self.model(input_tensor)
                loss, additional_information = self.criterion(outputs, target_tensor)
        else:
            outputs = self.model(input_tensor)
            loss, additional_information = self.criterion(outputs, target_tensor)

        # Backward pass and optimizer step (if in training mode)
        if train:
            if self.use_scaler:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Collect evaluation metrics and outputs
        self.evaluator.collect(loss.item(), additional_information, batch, outputs)

        return loss, additional_information

    def train_epoch(self, epoch: int) -> None:
        """
        Executes a single training epoch over the training dataset.

        Args:
            epoch (int): Current epoch number for logging and progress tracking.
        """

        self.model.train()
        progress_bar = tqdm(total=len(self.train_loader), desc=f'Epoch {epoch}', leave=False, mininterval=10)
        for batch in self.train_loader:
            self.process_batch(batch, train=True)
            progress_bar.update(1)
        progress_bar.close()
        self.evaluator.return_collection(print_output=True, train=True)

    def validate_epoch(self, epoch: int) -> None:
        """
        Executes a validation epoch without updating model weights.

        Args:
            epoch (int): Current epoch number for logging and progress tracking.

        Returns:
            avg_val_loss (float): Average validation loss over the epoch.
        """

        self.model.eval()
        progress_bar = tqdm(total=len(self.val_loader), desc=f'Validation {epoch}', leave=False, mininterval=10)
        with torch.no_grad():
            for batch in self.val_loader:
                self.process_batch(batch, train=False)
                progress_bar.update(1)
        progress_bar.close()
        avg_val_loss, _ = self.evaluator.return_collection(print_output=True, train=False, test=False, return_loss=True)

        # Return average validation loss
        return avg_val_loss

    def save_model(self, epoch: int) -> None:
        """
        Saves the current model state and weights for a given epoch.

        Args:
          epoch (int): Epoch number used to label the saved model file.
        """

        save_path = os.path.join(self.path, f'model_epoch_{epoch}.pth')
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f'Model saved at {save_path}')

    def train(self) -> None:
        """
        Runs the full training loop for the number of epochs defined in config.
        Includes training, periodic validation, scheduler stepping, and model saving.
        """

        for epoch in range(self.config['num_epochs']):
            self.train_epoch(epoch)

            # Perform validation at specified frequency
            if epoch % self.config['validation_frequency'] == 0:
                val_loss = self.validate_epoch(epoch)

                # Step the scheduler if it's a ReduceLROnPlateau or similar
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    self.scheduler.step(val_loss)
                    print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}, LR = {self.optimizer.param_groups[0]['lr']:.6f}")

                # Save model if validation loss improved
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model(epoch)


if __name__ == "__main__":
    raise NotImplementedError("This script is not intended to be run directly.")

