import numpy as np
import torch
from typing import Optional, Tuple, Dict, Union, List
import wandb
from collections import defaultdict
import logging


class InTrainingEvaluator:
    """
    Creates the model evaluator.

    Args:
        config (Dict): Configuration dictionary containing relevant parameters (e.g., radius for metric scaling).
        path (str): Path for saving and logging evaluation outputs.
    """

    def __init__(self, config: Dict, path: str) -> None:
        self.config = config
        self.path = path
        self.loss = []
        self.additional_losses = defaultdict(list)
        self.additional_information = defaultdict(list)
        self.epoch_counter = 0
        self.predictions_counter = 0

    def collect(
        self,
        loss: float,
        additional_loss_information: Dict,
        batch: List[torch.Tensor],
        outputs: torch.Tensor
    ) -> None:
        """
        Collects loss and additional information during training.

        Args:
            loss (float): The loss value for the current batch.
            additional_loss_information (dict): Additional loss information.
            batch (List[torch.Tensor]): Batch data, tuple of (input_tensor, target_tensor, indexes).
            outputs (torch.Tensor): Model outputs.
        """

        self.loss.append(loss)

        # Collect additional losses
        feature_losses = ['class_loss', 'position_loss', 'speed_loss', 'lat_acc_loss', 'tan_acc_loss', 'sin_angle_loss', 'cos_angle_loss']
        for feature in feature_losses:
            self.additional_losses[feature].append(additional_loss_information.get(feature, 0.0))

        target_tensor = batch[1]
        relevant_mask = target_tensor[:, :, 0] == 1

        outputs = outputs[relevant_mask]

        # Apply sigmoid and threshold to outputs
        outputs_binary = (torch.sigmoid(outputs[:, 0]) > 0.5).float()
        outputs_one_mask = outputs_binary == 1

        target_tensor_device = target_tensor[relevant_mask].to(outputs.device)

        # Get the number of agents that are 1 in the outputs_binary
        self.additional_information['count_acc'].append((outputs_binary == 1).sum().item()/(outputs_binary.shape[0] + 1e-6))

        # Calculate the mean and absolute differences for all features
        feature_names = ['position_x', 'position_y', 'speed', 'lat_acc', 'tan_acc', 'sin_angle', 'cos_angle']
        distances = {key: 0 for key in
                     ['pos_distance', 'pos_distance_ones', 'pos_distance_ones_veh', 'pos_distance_ones_vru', 'pos_distance_zeros'] + feature_names +
                     [f'{key}_ones' for key in feature_names] + [f'{key}_zeros' for key in feature_names]}

        if target_tensor_device.shape[0] > 0:
            diff = outputs[:, 1:] - target_tensor_device[:, 1:]
            for i, feature in enumerate(feature_names):
                distances[feature] = torch.mean(torch.abs(diff[:, i])).item()
            distances['pos_distance'] = torch.mean(torch.norm(diff[:, :2], dim=1)).item()

            if outputs_one_mask.sum() > 0:
                diff_ones = outputs[outputs_one_mask][:, 1:] - target_tensor_device[outputs_one_mask][:, 1:]
                for i, feature in enumerate(feature_names):
                    distances[f'{feature}_ones'] = torch.mean(torch.abs(diff_ones[:, i])).item()
                distances['pos_distance_ones'] = torch.mean(torch.norm(diff_ones[:, :2], dim=1)).item()

                # Mask for VRUs (feature 8 == 0 or 1) and vehicles (others)
                feature_class = target_tensor_device[outputs_one_mask][:, 8]
                vru_mask = (feature_class == 0) | (feature_class == 1)
                veh_mask = ~vru_mask

                # VRUs
                if vru_mask.sum() > 0:
                    distances['pos_distance_ones_vru'] = torch.mean(
                        torch.norm(diff_ones[vru_mask, :2], dim=1)).item()

                # Vehicles
                if veh_mask.sum() > 0:
                    distances['pos_distance_ones_veh'] = torch.mean(
                        torch.norm(diff_ones[veh_mask, :2], dim=1)).item()

            else:
                for i, feature in enumerate(feature_names):
                    distances[f'{feature}_ones'] = 0
                    distances['pos_distance_ones'] = 0
                    distances['pos_distance_ones_veh'] = 0
                    distances['pos_distance_ones_vru'] = 0

            if (~outputs_one_mask).sum() > 0:
                diff_zeros = outputs[~outputs_one_mask][:, 1:] - target_tensor_device[~outputs_one_mask][:, 1:]
                for i, feature in enumerate(feature_names):
                    distances[f'{feature}_zeros'] = torch.mean(torch.abs(diff_zeros[:, i])).item()
                distances['pos_distance_zeros'] = torch.mean(torch.norm(diff_zeros[:, :2], dim=1)).item()
            else:
                for i, feature in enumerate(feature_names):
                    distances[f'{feature}_zeros'] = 0
                distances['pos_distance_zeros'] = 0
        else:
            for key in distances.keys():
                distances[key] = 0

        for key, value in distances.items():
            self.additional_information[key].append(value)

    def return_collection(
            self,
            print_output: bool = False,
            train: bool = True,
            test: bool = False,
            return_loss: bool = False
    ) -> Optional[Union[Tuple[float, Dict[str, float]], None]]:
        """
        Returns the collected statistics and logs them.

        Args:
            print_output (bool, optional): Whether to print the output. Defaults to False.
            train (bool, optional): Whether it's training mode. Defaults to True.
            test (bool, optional): Whether it's testing mode. Defaults to False.
            return_loss (bool, optional): Whether to return the loss. Defaults to False.

        Returns:
            loss (float, optional): The mean loss if return_loss is True.
            return_dict (Dict, optional): The metric values if return_loss is True, containing:
                - 'loss': Mean training/validation/test loss.
                - <key>: Mean of each additional loss or performance metric.
                - For test mode: Distance-based statistics.
        """

        # Increment the epoch counter
        if train:
            self.epoch_counter += 1
        if test:
            self.predictions_counter += 1

        return_dict = {}

        # Compute mean loss
        loss = np.mean(self.loss).item()
        return_dict['loss'] = loss

        # Compute mean of additional losses
        for key, values in self.additional_losses.items():
            return_dict[key] = np.mean(values).item()

        # Compute mean and statistics of additional information
        for key, values in self.additional_information.items():
            return_dict[key] = np.mean(values).item()
            if key == "pos_distance_ones" and train is False and test is True:
                return_dict["mean_distance_ones [m]"] = np.mean(values) * self.config['radius']
                return_dict["median_distance_ones [m]"] = np.median(values) * self.config['radius']
                return_dict["quantile_90_distance_ones [m]"] = np.percentile(values, 90) * self.config['radius']
                return_dict["max_error_ones [m]"] = np.max(values) * self.config['radius']

        # Reset the stored losses and information
        self._reset()

        # Add 'val_' prefix if in validation mode
        if train is False and test is False:
            return_dict = {('val_' + key): value for key, value in return_dict.items()}
        if train is False and test is True:
            return_dict = {('test_' + key): value for key, value in return_dict.items()}

        # Print the output if requested
        if print_output:
            logger = logging.getLogger(__name__)
            logger.info(", ".join(f"{key}: {value:.4f}" for key, value in return_dict.items()))

        # Log the results to wandb
        if test is False:
            wandb.log(return_dict, step=self.epoch_counter)
        else:
            wandb.log(return_dict, step=self.epoch_counter+self.predictions_counter)

        if return_loss:
            return loss, return_dict

    def _reset(self) -> None:
        """Resets the stored losses and additional information."""
        self.loss.clear()
        self.additional_losses.clear()
        self.additional_information.clear()


if __name__ == "__main__":
    raise NotImplementedError("This script is not meant to be executed")
