import torch.nn as nn 
import torch
import torch.nn.functional as F
from typing import Tuple, Dict


class ClassPositionLoss(nn.Module):
    """
    Computes the binary classification loss and l1 loss for the position

    Args:
        weight_class (float): Weight for the classification loss.
        position_weight (float): Weight for the position loss.
    """

    def __init__(self, class_weight: float = 1.0, position_weight: float = 100.0) -> None:
        super().__init__()
        self.weight_class = class_weight
        self.weight_position = position_weight
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs (torch.Tensor): Model outputs of shape (batch_size, max_agents, num_features).
            targets (torch.Tensor): Ground truth tensor with the same shape as outputs.

        Returns:
            total_loss (torch.Tensor): Combined weighted classification and position loss.
            additional_information (Dict): Dictionary containing the individual weighted loss components:
                - 'class_loss': Weighted classification loss (float)
                - 'position_loss': Weighted position loss (float)
        """

        # Mask to select only relevant items where target_labels == 1
        target_labels = targets[:, :, 0]
        relevant_mask = target_labels == 1

        # Select predicted and target classifications for relevant samples
        relevant_pred_logit = outputs[:, :, 0][relevant_mask]
        relevant_target_logit = targets[:, :, 0][relevant_mask]

        # Compute loss only for relevant items
        # Classification loss
        if relevant_pred_logit.numel() > 0:
            class_loss = F.binary_cross_entropy_with_logits(relevant_pred_logit, relevant_target_logit,
                                                            reduction='mean')
        else:
            class_loss = torch.tensor(0.0, device=outputs.device)

        # Position loss
        # Compute predicted probabilities for position weighting (soft weights)
        # Use sigmoid to get probabilities from logits
        relevant_pred_probs = torch.sigmoid(relevant_pred_logit)

        # Select predicted and target positions for relevant samples
        relevant_pred_position = outputs[:, :, 1:][relevant_mask]
        relevant_target_position = targets[:, :, 1:][relevant_mask]

        # Compute position loss with soft weights
        if relevant_pred_position.numel() > 0:
            # Compute L2 loss without reduction to keep per-element loss
            position_loss = F.mse_loss(relevant_pred_position, relevant_target_position, reduction='none')
            # Mean over position dimensions (assuming positions are the last dimension)
            position_loss = position_loss.mean(dim=1)
            # Weight position loss by predicted probabilities (soft weights)
            weighted_position_loss = (position_loss * relevant_pred_probs).mean()
        else:
            weighted_position_loss = torch.tensor(0.0, device=outputs.device)

        # Total loss
        total_loss = self.weight_class * class_loss + self.weight_position * weighted_position_loss

        additional_information = {
            'class_loss': class_loss.item() * self.weight_class,
            'position_loss': weighted_position_loss.item() * self.weight_position
        }

        return total_loss, additional_information


class MultiFeatureLoss(nn.Module):
    """
    Computes the loss for all dynamic features: classification, position, speed, acceleration (lat/tan) and
    angle (sin/cos).

    Args:
        class_weight (float): Weight for the classification loss.
        position_weight (float): Weight for the position loss.
        speed_weight (float): Weight for the speed loss.
        acceleration_weight (float): Weight for the acceleration loss (applies to both lat/tan).
        angle_weight (float): Weight for the angle loss (sin/cos).
    """

    def __init__(self, class_weight: float = 10.0, position_weight: float = 1000.0,
                 speed_weight: float = 1.0, acceleration_weight: float = 1.0,
                 angle_weight: float = 1.0) -> None:

        super().__init__()
        self.weight_class = class_weight
        self.weight_position = position_weight
        self.weight_speed = speed_weight
        self.weight_acceleration = acceleration_weight
        self.weight_angle = angle_weight

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs (torch.Tensor): Model outputs of shape (batch_size, max_agents, num_features).
            targets (torch.Tensor): Ground truth tensor with the same shape as outputs.

        Returns:
            total_loss (torch.Tensor): Combined feature loss.
            additional_information (Dict): Dictionary containing the individual weighted loss components:
                - 'class_loss': Weighted classification loss (float)
                - 'position_loss': Weighted position loss (float)
                - 'speed_loss': Weighted speed loss (float)
                - 'acceleration_loss': Weighted acceleration loss (lateral and tangential, float)
                - 'angle_loss': Weighted angle loss (sin and cos, float)
        """

        # Feature indices
        CLASS_IDX = 0
        POSITION_IDX = slice(1, 3)
        SPEED_IDX = 3
        ACC_IDX = slice(4, 6)
        ANGLE_IDX = slice(6, 8)

        # Mask to select only relevant items where target_labels == 1
        target_labels = targets[:, :, CLASS_IDX]
        relevant_mask = target_labels == 1

        # Select predicted and target classifications for relevant samples
        relevant_pred_logit = outputs[:, :, CLASS_IDX][relevant_mask]
        relevant_target_logit = targets[:, :, CLASS_IDX][relevant_mask]

        # Compute loss only for relevant items
        # Classification loss
        if relevant_pred_logit.numel() > 0:
            class_loss = F.binary_cross_entropy_with_logits(relevant_pred_logit, relevant_target_logit, reduction='mean')
        else:
            class_loss = torch.tensor(0.0, device=outputs.device)

        # Position loss
        # Soft weighting with additional weighting for large deviations
        # Compute predicted probabilities for position weighting (soft weights)
        # Use sigmoid to get probabilities from logits
        relevant_pred_probs = torch.sigmoid(relevant_pred_logit)

        # Select predicted and target positions for relevant samples
        relevant_pred_position = outputs[:, :, POSITION_IDX][relevant_mask]
        relevant_target_position = targets[:, :, POSITION_IDX][relevant_mask]

        # Compute position loss with soft weights
        if relevant_pred_position.numel() > 0:
            # Compute L2 loss without reduction to keep per-element loss
            # Mean over position dimensions (assuming positions are the last dimension)
            position_loss = F.mse_loss(relevant_pred_position, relevant_target_position, reduction='none').mean(
                dim=1)
            # Weight position loss by predicted probabilities (soft weights)
            weighted_position_loss = (position_loss * relevant_pred_probs).mean()
        else:
            weighted_position_loss = torch.tensor(0.0, device=outputs.device)

        # Speed loss
        # Select predicted and target speeds for relevant samples
        relevant_pred_speed = outputs[:, :, SPEED_IDX][relevant_mask]
        relevant_target_speed = targets[:, :, SPEED_IDX][relevant_mask]
        # Compute MSE loss for speed only for relevant targets
        if relevant_pred_speed.numel() > 0:
            speed_loss = F.mse_loss(relevant_pred_speed, relevant_target_speed, reduction='mean')
        else:
            speed_loss = torch.tensor(0.0, device=outputs.device)

        # Acceleration loss (lateral and tangential)
        # Select predicted and target accelerations for relevant samples
        relevant_pred_acc = outputs[:, :, ACC_IDX][relevant_mask]
        relevant_target_acc = targets[:, :, ACC_IDX][relevant_mask]

        if relevant_pred_acc.numel() > 0:
            # Compute Smooth L1 loss for lateral and tangential acceleration (robust to outliers)
            lat_acc_loss = F.smooth_l1_loss(relevant_pred_acc[:, 0], relevant_target_acc[:, 0], reduction='mean')
            tan_acc_loss = F.smooth_l1_loss(relevant_pred_acc[:, 1], relevant_target_acc[:, 1], reduction='mean')
        else:
            lat_acc_loss = torch.tensor(0.0, device=outputs.device)
            tan_acc_loss = torch.tensor(0.0, device=outputs.device)

        # Angle loss (sin and cos)
        # Select predicted and target heading angles (sin and cos) for relevant samples
        relevant_pred_angle = outputs[:, :, ANGLE_IDX][relevant_mask]
        relevant_target_angle = targets[:, :, ANGLE_IDX][relevant_mask]

        # Compute separate MSE loss for sine and cosine components of the angle
        if relevant_pred_angle.numel() > 0:
            sin_angle_loss = F.mse_loss(relevant_pred_angle[:, 0], relevant_target_angle[:, 0])
            cos_angle_loss = F.mse_loss(relevant_pred_angle[:, 1], relevant_target_angle[:, 1])
        else:
            sin_angle_loss = torch.tensor(0.0, device=outputs.device)
            cos_angle_loss = torch.tensor(0.0, device=outputs.device)

        # Enforce normalization on angle vector (unit circle constraint)
        # Penalizes predictions that deviate from the unit circle (i.e., sin² + cos² ≠ 1)
        unit_circle_loss = F.mse_loss(relevant_pred_angle[:, 0] ** 2 + relevant_pred_angle[:, 1] ** 2, torch.ones_like(relevant_pred_angle[:, 0]))

        # Total loss
        total_loss = (self.weight_class * class_loss +
                      self.weight_position * weighted_position_loss +
                      self.weight_speed * speed_loss +
                      self.weight_acceleration * (lat_acc_loss + tan_acc_loss) +
                      self.weight_angle * (sin_angle_loss + cos_angle_loss) +
                      (self.weight_angle / 10) * unit_circle_loss)

        additional_information = {
            'class_loss': class_loss.item() * self.weight_class,
            'position_loss': weighted_position_loss.item() * self.weight_position,
            'speed_loss': speed_loss.item() * self.weight_speed,
            'lat_acc_loss': lat_acc_loss.item() * self.weight_acceleration,
            'tan_acc_loss': tan_acc_loss.item() * self.weight_acceleration,
            'sin_angle_loss': sin_angle_loss.item() * self.weight_angle,
            'cos_angle_loss': cos_angle_loss.item() * self.weight_angle,
        }

        return total_loss, additional_information


if __name__ == "__main__":
    raise NotImplementedError("This script is not meant to be executed")
