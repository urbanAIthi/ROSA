import logging
import os
import torch
import matplotlib.pyplot as plt
import geopandas as gpd
from typing import Tuple, Any, List
from utils.occupancy_evaluation_utils import transform_geojson_to_utm


class Visualizer:
    """
    Visualizes model results for a given roundabout scenario based on test data.

    Args:
        roundabout_scenario (str): Name or identifier of the roundabout scenario.
        radius (float): Scaling factor used to convert normalized coordinates to metric values.
        center_point (Tuple[float, float]): UTM coordinates (x, y) used as the origin for normalization.
        log_dir (str): Directory to save the visualizations and logs.
        logger (logging.Logger): Logger instance used to log status and debugging information.
    """

    def __init__(self, roundabout_scenario: str, radius: float, center_point: Tuple[float, float], log_dir: str, logger: logging.Logger) -> None:
        self.roundabout_scenario = roundabout_scenario
        self.radius = radius
        self.center_point = center_point
        self.path = log_dir
        self.logger = logger

    def _renormalize_positions(self, data: torch.Tensor, mask: torch.Tensor) -> List[Tuple[float, float]]:
        """
        Helper function to filter agents and renormalize positions.

        Args:
            data (torch.Tensor): A tensor containing position data.
            mask (torch.Tensor): Mask that defines valid positions to consider.

        Returns:
            positions (list): List of tuples containing the renormalized positions (x, y).
        """

        data_valid = data[mask]
        positions = []
        for i in range(data_valid.shape[0]):
            # Renormalize x and y positions based on radius and center_point
            x = data_valid[i, 1] * self.radius + self.center_point[0]
            y = data_valid[i, 2] * self.radius + self.center_point[1]
            if x != 0 and y != 0:
                positions.append((x.item(), y.item()))
        return positions

    def log_positions(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
        """
            Logs ground truth and predicted positions for each sample and agent in a batch.

        Args:
            input_tensor (torch.Tensor): Input sequence tensor of shape (batch_size, max_agents, sequence_length, num_features), containing past and current agent data.
            target_tensor (torch.Tensor): Tensor of shape (batch_size, max_agents, num_features), containing ground truth data for the prediction step.
            output_tensor (torch.Tensor): Tensor of shape (batch_size, max_agents, num_features), containing model predictions for the prediction step.
        """

        batch_size = input_tensor.shape[0]
        for sample_idx in range(batch_size):
            target = target_tensor[sample_idx]
            output = output_tensor[sample_idx]
            input = input_tensor[sample_idx]

            # Masks for filtering valid positions
            mask_logit1 = (target[:, 0] == 1) & (target[:, 1] != 0) & (target[:, 2] != 0)
            mask_other = ((target[:, 0] != 1) & (target[:, 0] != 0)) & (target[:, 1] != 0) & (target[:, 2] != 0)

            if not mask_logit1.any() and not mask_other.any():
                continue

            # Renormalize positions for input, target, and output
            input_positions = self._renormalize_positions(input[-1], mask_logit1)
            target_positions = self._renormalize_positions(target, mask_logit1)
            output_positions = self._renormalize_positions(output.to(dtype=torch.float32), mask_logit1)

            # Log the positions
            self.logger.info(f"Sample {sample_idx} input positions: {input_positions}")
            self.logger.info(f"Sample {sample_idx} target positions: {target_positions}")
            self.logger.info(f"Sample {sample_idx} output positions: {output_positions}")

    def visualize_batch(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor, final_prediction: torch.Tensor, batch_idx: int, shapefile_path: str) -> None:
        """
        Visualizes all samples per batch with target and predicted positions on a map.

        Args:
            input_tensor (torch.Tensor): Input sequence tensor of shape (batch_size, max_agents, sequence_length, num_features), containing past and current agent data.
            target_tensor (torch.Tensor): Tensor of shape (batch_size, max_agents, num_features), containing ground truth data for the final prediction step.
            final_prediction (torch.Tensor): Tensor of shape (batch_size, max_agents, num_features), containing model predictions for the final prediction step.
            batch_idx (int): Index of the current batch.
            shapefile_path (str): Path to the shapefile for the background map.
        """

        os.makedirs(self.path, exist_ok=True)
        batch_save_dir = os.path.join(self.path, f"batch_{batch_idx}")
        os.makedirs(batch_save_dir, exist_ok=True)

        # Load shapefile for background map
        shapefile = gpd.read_file(shapefile_path)
        gdf_polygons_veh = transform_geojson_to_utm(os.path.join(self.roundabout_scenario, 'polygons', 'veh_polygons.json'))
        gdf_polygons_vru = transform_geojson_to_utm(os.path.join(self.roundabout_scenario, 'polygons', 'vru_polygons.json'))

        batch_size = input_tensor.shape[0]

        for sample_idx in range(batch_size):
            target = target_tensor[sample_idx]
            output = final_prediction[sample_idx]

            # Masks for filtering valid positions
            mask_logit1 = (target[:, 0] == 1) & (target[:, 1] != 0) & (target[:, 2] != 0)
            mask_other = ((target[:, 0] != 1) & (target[:, 0] != 0)) & (target[:, 1] != 0) & (target[:, 2] != 0)

            # Skip if no valid positions found
            if not mask_logit1.any() and not mask_other.any():
                continue

            plt.figure(figsize=(10, 10))
            ax = shapefile.plot(color='gray', edgecolor='black', figsize=(10, 10))
            gdf_polygons_veh.plot(ax=ax, color='orange', edgecolor='red', alpha=0.5)
            gdf_polygons_vru.plot(ax=ax, color='blue', edgecolor='red', alpha=0.5)

            # Plot target and output positions
            ax.scatter([], [], color='blue', label='Target')
            ax.scatter([], [], color='red', label='Output')
            for x, y in self._renormalize_positions(target, mask_logit1):
                ax.scatter(x, y, color='blue')
            for x, y in self._renormalize_positions(output.to(dtype=torch.float32), mask_logit1):
                ax.scatter(x, y, color='red', alpha=0.5)

            # Plot other logits if any
            if mask_other.any():
                ax.scatter([], [], color='green', label='Other logits')
                for x, y in self._renormalize_positions(target, mask_other):
                    ax.scatter(x, y, color='green')

            # Save the visualization as a PNG file
            plt.legend()
            plt.xlabel("X-Position")
            plt.ylabel("Y-Position")
            plt.title(f"Batch {batch_idx}, Sample {sample_idx}")
            file_path = os.path.join(batch_save_dir, f"batch_{batch_idx}_sample_{sample_idx}.png")
            plt.savefig(file_path)
            plt.close()

        self.logger.info(f'Visualizations saved for batch {batch_idx}')


if __name__ == "__main__":
    raise NotImplementedError("This script is not intended to be run directly.")