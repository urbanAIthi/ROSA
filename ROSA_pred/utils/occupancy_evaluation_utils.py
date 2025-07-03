import json
import os
import geopandas as gpd
import pyproj
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Set
from shapely.geometry import Point, shape


def transform_geojson_to_utm(path: Union[str, Path], epsg_code: int = 32632) -> gpd.GeoDataFrame:
    """
    Loads a GeoJSON file, transforms it to the UTM coordinate system, and returns as a GeoDataFrame.

    Args:
        path (str): Path to the GeoJSON file.
        epsg_code (int): EPSG code for the UTM projection.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with transformed polygons.
    """

    path = Path(path)
    # Create a Coordinate Reference System (CRS) object for the specified UTM projection
    utm_crs = pyproj.CRS.from_epsg(epsg_code)

    with path.open("r", encoding="utf-8") as fp:
        geojson = json.load(fp)

    feats = []
    # Set up a transformer to convert from WGS84 (lat/lon) to UTM
    transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    # Iterate through each feature in the GeoJSON
    for feat in geojson["features"]:
        # Convert the geometry into a shapely Polygon object
        poly = shape(feat["geometry"])
        # Apply coordinate transformation to all exterior points of the polygon
        coords = [transformer.transform(x, y) for x, y in poly.exterior.coords]
        # Update the feature geometry with the transformed UTM coordinates
        feat["geometry"] = shape({"type": "Polygon", "coordinates": [coords]}).__geo_interface__
        # Add the transformed feature to the list
        feats.append(feat)

    # Create and return a GeoDataFrame from the transformed features using the UTM CRS
    return gpd.GeoDataFrame.from_features(feats, crs=utm_crs)


class OccupancyEvaluator:
    """
    Evaluates agent positions and occupancy within predefined zones for a specific roundabout scenario..

    Args:
        roundabout_scenario (str): Identifier or name of the roundabout scenario to be evaluated.
        radius (float): Scaling factor used to convert normalized coordinates to metric values.
        center_point (Tuple[float, float]): UTM coordinates (x, y) used as the origin for normalization.
    """

    def __init__(self, roundabout_scenario: str, radius: float, center_point: Tuple[float, float]) -> None:
        self.roundabout_scenario = roundabout_scenario
        self.radius = radius
        self.center_point = center_point
        self.data_entries: List[list] = []

        # Load and prepare polygons for the zones
        self._veh_polys = transform_geojson_to_utm(os.path.join(self.roundabout_scenario, 'polygons', 'veh_polygons.json'))
        self._vru_polys = transform_geojson_to_utm(os.path.join(self.roundabout_scenario, 'polygons', 'vru_polygons.json'))

    def evaluate_occupancy( self, target_tensor: torch.Tensor, output_tensor: torch.Tensor
    ) -> Tuple[Dict[int, Dict[str, Dict[str, int]]], Dict[str, Dict[str, int]]]:
        """
        Evaluates zone occupancy for VRUs and vehicles by comparing target and predicted positions.

        For each sample in the batch, this function computes true positives (tp), false positives (fp),
        false negatives (fn), and true negatives (tn) based on whether predicted and target positions
        fall into predefined polygon zones.

        Args:
            target_tensor (torch.Tensor): Ground truth tensor of shape (batch_size, max_agents, num_features).
            output_tensor (torch.Tensor): Predicted output tensor of the same shape.

        Returns:
            sample_results (Dict): Dictionary with per-sample occupancy metrics.
            batch_results (Dict): Aggregated occupancy metrics across all valid samples.
        """

        sample_results = {}
        batch_size = output_tensor.size(0)

        # Initialize total metric counts for both classes
        total_metrics = {
            "VRU": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
            "Vehicle": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        }

        for sample_idx in range(batch_size):
            sample_metrics = {
                "VRU": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
                "Vehicle": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            }

            target_sample = target_tensor[sample_idx]
            output_sample = output_tensor[sample_idx]

            # Iterate over VRUs and Vehicles
            for entity, condition, polygons in zip(
                ["VRU", "Vehicle"],
                [lambda x: x <= 1, lambda x: x >= 2], # Class conditions
                [self._vru_polys, self._veh_polys]): # Corresponding polygons

                # Filter valid agents based on mask in feature channel 0
                mask = (target_sample[:, 0] == 1) & condition(target_sample[:, 8])
                if not mask.any():
                    continue  # Skip sample if no valid agents

                target_valid = target_sample[mask]
                output_valid = output_sample[mask].to(torch.float32)

                # Renormalize (x, y) positions for ground truth and prediction
                pos_target = self._renormalize_positions(target_valid)
                pos_output = self._renormalize_positions(output_valid)

                # Determine which zones are occupied by target and predicted positions
                occupied_target_zones = self.get_occupied_zones(pos_target, polygons)
                occupied_output_zones = self.get_occupied_zones(pos_output, polygons)

                # Compare zone occupancy
                for zone_idx in range(len(polygons)):
                    is_target = zone_idx in occupied_target_zones
                    is_output = zone_idx in occupied_output_zones

                    if is_target and is_output:
                        key = "tp"
                    elif not is_target and is_output:
                        key = "fp"
                    elif is_target and not is_output:
                        key = "fn"
                    else:
                        key = "tn"

                    sample_metrics[entity][key] += 1

            # Store per-sample metrics
            sample_results[sample_idx] = sample_metrics

            # Accumulate batch metrics
            for e in total_metrics:
                for k in total_metrics[e]:
                    total_metrics[e][k] += sample_metrics[e][k]

            batch_results = total_metrics

        return sample_results, batch_results

    def evaluate_entry(self, target_tensor: torch.Tensor, output_tensor: torch.Tensor, batch_idx: int) -> List[List[Any]]:
        """
        Evaluates whether any vehicles or VRUs block a specific roundabout entry, i.e., enter a specific polygon zone.

        For each sample in the batch, this function checks if the target or predicted positions
        of entities (vehicles or VRUs) are inside predefined entry polygons. The results are stored
        as "r" (red: in polygon) or "g" (green: not in polygon).

        Args:
            target_tensor (torch.Tensor): Ground truth tensor of shape (batch_size, max_agents, num_features).
            output_tensor (torch.Tensor): Predicted output tensor of same shape.
            batch_idx (int): Current batch index used for tracking/logging.

        Returns:
            self.data_entries (list): The updated `self.data_entries` list with appended results per sample:
            [batch_idx, sample_idx, target_vehicle_state, target_vru_state, output_vehicle_state, output_vru_state]
        """

        # Mapping of class types to corresponding polygon file and condition for selection
        polygon_files = {
            "veh": (os.path.join(self.roundabout_scenario, 'polygons', 'veh_entry_polygon.json'), lambda x: x >= 2),
            "vru": (os.path.join(self.roundabout_scenario, 'polygons', 'vru_entry_polygon.json'), lambda x: x <= 1)
        }

        # Load and transform polygons into UTM coordinates
        polygons = {key: transform_geojson_to_utm(file) for key, (file, _) in polygon_files.items()}

        batch_size = output_tensor.size(0)
        for sample_idx in range(batch_size):
            target_sample = target_tensor[sample_idx]
            output_sample = output_tensor[sample_idx]

            states_output = {}
            states_target = {}

            # Evaluate VRUs and Vehicles separately
            for key, (_, condition) in polygon_files.items():
                # Filter valid agents of current type
                mask = (target_sample[:, 0] == 1) & condition(target_sample[:, 8])
                if not mask.any():
                    states_output[key] = states_target[key] = 'g'
                    continue

                target_valid = target_sample[mask]
                output_valid = output_sample[mask].to(torch.float32)

                # Convert normalized positions back to original coordinates
                pos_target = self._renormalize_positions(target_valid)
                pos_output = self._renormalize_positions(output_valid)

                # Assign state based on whether any point is in polygon
                states_target[key] = 'r' if self.is_in_polygon(pos_target, polygons[key]) else 'g'
                states_output[key] = 'r' if self.is_in_polygon(pos_output, polygons[key]) else 'g'

            # Append evaluation result for current sample
            self.data_entries.append([
                batch_idx, sample_idx,
                states_target["veh"], states_target["vru"],
                states_output["veh"], states_output["vru"]
            ])

        return self.data_entries

    def _renormalize_positions(self, data: torch.Tensor) -> torch.Tensor:
        """
        Helper function to renormalize positions.

        Args:
            data (torch.Tensor): A tensor containing position data.

        Returns:
            torch.Tensor: Reprojected positions in UTM coordinates.
        """
        x = data[:, 1] * self.radius + self.center_point[0]
        y = data[:, 2] * self.radius + self.center_point[1]
        return torch.stack((x, y), dim=-1)

    def get_occupied_zones(self, positions: List[tuple[float, float]], polygons: gpd.GeoDataFrame) -> Set[int]:

        """
        Returns the indices of polygons occupied by at least one of the given positions.

        Args:
            positions (list): A list of (x, y) positions.
            polygons (gpd.GeoDataFrame): A GeoDataFrame with polygon geometries.

        Returns:
            occupied (set): Indices of polygons that contain at least one point.
        """

        occupied = set()
        for pos in positions:
            point = Point(float(pos[0]), float(pos[1]))
            for idx, poly in enumerate(polygons.geometry):
                if poly.contains(point):
                    occupied.add(idx)
        return occupied

    def is_in_polygon(self, positions: List[Tuple[float, float]], polygons: gpd.GeoDataFrame) -> bool:
        """
        Check if the given position is inside of the polygons.

        Args:
            positions (list): A list of (x, y) positions.
            polygons (gpd.GeoDataFrame): A GeoDataFrame with polygon geometries.

        Returns:
            bool: True if the position is inside any polygon, else False.
        """
        for pos in positions:
            point = Point(float(pos[0]), float(pos[1]))
            if any(poly.contains(point) for poly in polygons.geometry):
                return True
        return False


if __name__ == "__main__":
    raise NotImplementedError("This script is not intended to be run directly.")


