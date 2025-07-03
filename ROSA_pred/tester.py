import pandas as pd
import logging
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.intraining_evaluation import InTrainingEvaluator
from utils.visualization_utils import Visualizer
from utils.occupancy_evaluation_utils import OccupancyEvaluator
from utils.setup_utils import set_seed

torch.backends.cudnn.benchmark = True

class Tester:
    """
    Tester class to evaluate a trained model on a test dataset.

    Args:
        model (nn.Module): The trained PyTorch model for prediction.
        test_loader (DataLoader): DataLoader providing the test data.
        criterion (MultiFeatureLoss): Loss function used for evaluation.
        device (torch.device): Device on which to perform training (CPU or CUDA).
        evaluator (InTrainingEvaluator): Evaluator object to track and log evaluation metrics.
        roundabout_scenario (str): Name or identifier of the selected roundabout scenario from config.
        path (str): Path for logs/visuals.
        directory (str): Directory to save results.
        prediction_len (int): Number of autoregressive prediction steps.
        radius (float): Scaling factor used to convert normalized coordinates to metric values.
        center_point (Tuple[float, float]): UTM coordinates (x, y) used as the origin for normalization.
        shapefile_path (str): Path to shapefile for visualization.
        visualization_mode (bool): Enables visualization output.
        logger (logging.Logger): Logger instance for status messages.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        evaluator: InTrainingEvaluator,
        roundabout_scenario: str,
        path: str,
        directory: str,
        prediction_len: int,
        radius: int,
        centerpoint: Tuple[float, float],
        shapefile_path: str,
        visualization_mode: bool,
        logger: logging.Logger
    ) -> None:
        self.model = model.to(device)
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.evaluator = evaluator
        self.path = path
        self.directory = directory
        self.prediction_len = prediction_len
        self.logger = logger

        self.roundabout_scenario = roundabout_scenario
        self.radius = radius
        self.center_point = centerpoint

        self.shapefile_path = shapefile_path
        self.excel_path = f"{self.directory}/occupancy_states.xlsx"

        # Visualizations
        self.visualization_mode = visualization_mode
        if self.visualization_mode:
            self.visualizer = Visualizer(
                roundabout_scenario=roundabout_scenario,
                radius=self.radius,
                center_point=self.center_point,
                log_dir=self.path,
                logger=self.logger
            )

        # Occupancy Evaluations
        self.occ_evaluator = OccupancyEvaluator(
            roundabout_scenario=roundabout_scenario,
            radius=self.radius,
            center_point=self.center_point
        )

        self.data_entries = []

        # Initialize GradScaler for mixed precision if needed
        self.scaler = torch.amp.GradScaler()

    def process_batch(
            self,
            batch_idx: int,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[float, Dict, Dict, Dict]:
        """
        Processes a single test batch: performs autoregressive prediction, computes loss and evaluation metrics.

        Args:
            batch_idx (int): Index of the current batch in the test loop.
            batch (Tuple[Tensor, Tensor, Tensor]): A tuple containing:
               - input_tensor (Tensor): Shape (batch_size, sequence_length, max_agents, num_features).
               - target_tensor (Tensor): Shape (batch_size, sequence_length, max_agents, num_features).
               - sample_indices (Tensor): Identifiers for each sample in the batch.

        Returns:
            loss.item() (float): Scalar loss value for the batch.
            additional_information (Dict): Dictionary with individual loss component values.
            sample_results (Dict): Occupancy evaluation results per sample.
            batch_results (Dict): Aggregated evaluation metrics per batch.
        """

        self.logger.info(f"Processing Batch {batch_idx}")

        input_tensor, target_tensor, indexes = batch
        input_tensor, target_tensor = input_tensor.to(self.device), target_tensor.to(self.device)

        # Set the first 5 timesteps in the input tensor to zero (data masking)
        input_tensor[:, :5, :, :] = 0

        predictions = []
        # Autoregressive prediction
        for timestep in range(self.prediction_len):
            self.logger.info(f"Processing timestep {timestep} of prediction_len {self.prediction_len}")

            # Perform forward pass using mixed precision
            with torch.amp.autocast(device_type=self.device.type):
                output_tensor = self.model(input_tensor)

            predictions.append(output_tensor)  # Shape: (batch_size, max_agents, num_features)

            # Log positions per sample
            #if self.visualization_mode:
                #self.visualizer.log_positions(input_tensor, target_tensor, output_tensor)

            # Update input tensor for the next step (autoregressive prediction)
            # Remove the oldest timestep from input
            input_tensor = input_tensor[:, 1:, :, :]  # Shape: (batch_size, sequence_len-1, max_agents, num_features)

            # Apply sigmoid to the first feature (confidence)
            output_tensor[:, :, 0].sigmoid_()
            # Replace static features from index 8 onward with the ground truth (to maintain context)
            output_tensor[:, :, 8:] = target_tensor[:, :, 8:]
            #output_with_sigmoid[:, :, 3:] = target_tensor[:, :, 3:]

            # Reshape to match input format: add a time dimension
            # Append new timestep to the input tensor
            input_tensor = torch.cat((input_tensor, output_tensor.unsqueeze(1)), dim=1)

        # Evaluate final timestep prediction
        final_prediction = predictions[-1]
        # Compute total loss and additional metrics
        loss, additional_information = self.criterion(final_prediction, target_tensor)
        # Collect evaluation statistics
        self.evaluator.collect(loss.item(), additional_information, batch, final_prediction)
        # Visualize the batch predictions
        if self.visualization_mode:
            self.visualizer.visualize_batch(input_tensor, target_tensor, final_prediction, batch_idx, self.shapefile_path)
        # Compute occupancy evaluation metrics
        sample_results, batch_results = self.occ_evaluator.evaluate_occupancy(target_tensor, final_prediction)
        # Evaluate and track roundabout entry states
        self.data_entries = self.occ_evaluator.evaluate_entry(target_tensor, final_prediction, batch_idx)

        return loss.item(), additional_information, sample_results, batch_results


    def test(self) -> None:
        """
        Executes the full test routine across the test dataset, calculates and logs occupancy metrics for VRUs and vehicles.
        """

        set_seed(42)
        self.model.eval()
        progress_bar = tqdm(total=len(self.test_loader), desc='Testing', leave=False, mininterval=10)

        # Initialize occupancy metric counters for VRUs and vehicles
        total_tp_vru, total_fp_vru, total_fn_vru, total_tn_vru = 0, 0, 0, 0
        total_tp_vehicle, total_fp_vehicle, total_fn_vehicle, total_tn_vehicle = 0, 0, 0, 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # Process the batch and get evaluation results
                _, _, sample_results, batch_results = self.process_batch(batch_idx, batch)
                self.logger.info(f'Results per sample in batch {batch_idx}: {sample_results}')
                self.logger.info(f'Overall results in batch {batch_idx}: {batch_results}')

                # Accumulate classification statistics across batches
                total_tp_vru += batch_results["VRU"]["tp"]
                total_fp_vru += batch_results["VRU"]["fp"]
                total_fn_vru += batch_results["VRU"]["fn"]
                total_tn_vru += batch_results["VRU"]["tn"]

                total_tp_vehicle += batch_results["Vehicle"]["tp"]
                total_fp_vehicle += batch_results["Vehicle"]["fp"]
                total_fn_vehicle += batch_results["Vehicle"]["fn"]
                total_tn_vehicle += batch_results["Vehicle"]["tn"]

                progress_bar.update(1)

            # Compute final occupancy metrics for VRUs
            final_recall_vru = total_tp_vru / max(total_tp_vru + total_fn_vru, 1e-6)
            final_precision_vru = total_tp_vru / max(total_tp_vru + total_fp_vru, 1e-6)
            final_f1_score_vru = 2 * (final_precision_vru * final_recall_vru) / max(final_precision_vru + final_recall_vru, 1e-6)
            final_accuracy_vru = (total_tp_vru + total_tn_vru) / max(total_tp_vru + total_fp_vru + total_fn_vru + total_tn_vru, 1e-6)

            # Compute final occupancy metrics for vehicles
            final_recall_vehicle = total_tp_vehicle / max(total_tp_vehicle + total_fn_vehicle, 1e-6)
            final_precision_vehicle = total_tp_vehicle / max(total_tp_vehicle + total_fp_vehicle, 1e-6)
            final_f1_score_vehicle = 2 * (final_precision_vehicle * final_recall_vehicle) / max(final_precision_vehicle + final_recall_vehicle, 1e-6)
            final_accuracy_vehicle = (total_tp_vehicle + total_tn_vehicle) / max(total_tp_vehicle + total_fp_vehicle + total_fn_vehicle + total_tn_vehicle, 1e-6)

            self.logger.info("===== PREDICTION LEN =====")
            self.logger.info(self.prediction_len)
            self.logger.info("===== FINAL VRU METRICS =====")
            self.logger.info(f"TP: {total_tp_vru}, FP: {total_fp_vru}, FN: {total_fn_vru}, TN: {total_tn_vru}")
            self.logger.info(f"Recall: {final_recall_vru:.4f}, Precision: {final_precision_vru:.4f}, "
                             f"F1: {final_f1_score_vru:.4f}, Accuracy: {final_accuracy_vru:.4f}")

            self.logger.info("===== FINAL VEHICLE METRICS =====")
            self.logger.info(f"TP: {total_tp_vehicle}, FP: {total_fp_vehicle}, FN: {total_fn_vehicle}, TN: {total_tn_vehicle}")
            self.logger.info(f"Recall: {final_recall_vehicle:.4f}, Precision: {final_precision_vehicle:.4f}, "
                             f"F1: {final_f1_score_vehicle:.4f}, Accuracy: {final_accuracy_vehicle:.4f}")

            # Save per-sample prediction states for roundabout entries to Excel
            df = pd.DataFrame(self.data_entries, columns=[
                'batch_idx', 'sample_idx',
                'state_target_veh', 'state_target_vru',
                'state_output_veh', 'state_output_vru'
            ])

            try:
                with pd.ExcelWriter(self.excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    df.to_excel(writer, sheet_name=f'Prediction_{self.prediction_len}', index=False)
            except FileNotFoundError:
                with pd.ExcelWriter(self.excel_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name=f'Prediction_{self.prediction_len}', index=False)

            progress_bar.close()

            _, return_dict = self.evaluator.return_collection(print_output=True, train=False, test=True, return_loss=True)
            self.logger.info(return_dict)


if __name__ == "__main__":
    raise NotImplementedError("This script is not intended to be run directly.")