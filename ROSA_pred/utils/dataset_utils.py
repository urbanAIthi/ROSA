import os
import math
import torch
import random
import pandas as pd
from tqdm import tqdm
from typing import Optional, Tuple, List, Union
from torch.utils.data import Dataset
from einops import rearrange
from collections import Counter


class SequenceDataset(Dataset):
    """
    Preprocesses the trajectory dataset for training, validation, testing.

    Args:
        dataset_path (str): Path to the dataset folder
        sequence_len (int): Number of historical timesteps to use as input sequence.
        max_agents (int): Maximum number of agents considered per scene.
        prediction_len (int): Number of future timesteps to predict per prediction step.
        max_prediction (int): Maximum number of future predictions to consider per sample.
        min_timesteps_seen (int): Minimum number of past timesteps an agent must have been seen.
        split (tuple): Tuple specifying start and end index to split the dataset (e.g. for train/test split).
        radius (int): Radius (in pixels or meters) around the centerpoint to crop the scene and normalize data.
        centerpoint (tuple): Center point (x, y) to crop the scene and normalize data.
        loop (Optional[int]): Optional limit for how many files/samples to load.
    """

    def __init__(self, dataset_path: List[str], sequence_len: int = 8, max_agents: int = 100,
                 prediction_len: int = 1, max_prediction: int = 5, min_timesteps_seen: int = 3,
                 split: tuple = None, radius: int = None, centerpoint: Tuple[int, int] = None, loop: Optional[int] = None) -> None:

        if split is not None:
            assert len(split) == len(dataset_path), "Split must have the same length as the dataset_path"
        else:
            split = [None] * len(dataset_path)

        self.dataset = None
        for path, split in tqdm(zip(dataset_path, split), desc='Loading datasets'):
            name = path.split('/')[-1]
            current_dataset = pd.read_pickle(os.path.join(path, 'dataset.pkl'))
            current_dataset = current_dataset.reset_index(drop=True)
            current_dataset['dataset_name'] = name

            # Only keep a split of the dataset for training/validation/testing
            current_dataset = self._split_dataset(current_dataset, split)

            # Only keep the loop specified
            if loop is not None:
                current_dataset = current_dataset[current_dataset['loop'] == loop]

            if self.dataset is None:
                self.dataset = current_dataset
            else:
                self.dataset = pd.concat([self.dataset, current_dataset], ignore_index=True)

        self.sequence_len = sequence_len
        self.max_agents = max_agents
        self.max_prediction = max_prediction
        self.prediction_len = prediction_len
        self.min_timesteps_seen = min_timesteps_seen

        if radius is not None:
            self.radius = radius
        if centerpoint is not None:
            self.center_point = centerpoint

        # Encode agent types
        self.type_encoding = {
            "Pedestrian": 0,
            "Bicycle": 1,
            "Motorcycle": 2,
            "Car": 3,
            "Medium Vehicle": 4,
            "Heavy Vehicle": 5,
            "Trailer": 6,
            "Bus": 7
        }

        self.max_agents_counter = 0
        self._compute_mean_std() # Compute mean and standard deviation values for normalization
        self._get_allowed_indexes()  # Create self.allowed_indexes
        self._create_inputs()  # Create self.input_tensors
        self._create_targets()  # Create self.target_tensors


    def __len__(self) -> int:
        return len(self.allowed_indexes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Creates input and target tensors.

        Args:
            idx (int): Index of the dataset to retrieve

        Returns:
            input_tensor (torch.Tensor): Input tensor of shape (sequence_len, max_agents, num_features)
            target_tensor (torch.Tensor): Target tensor of shape (max_agents, num_features)
            index (int): Index of the dataset
        """

        index = self.allowed_indexes[idx]

        # Get the sequence datapoint IDs
        sequence_indexes = list(range(index - self.sequence_len + 1, index + 1))
        sequence_datapoint_ids = [self.dataset.loc[i, 'id'] for i in sequence_indexes] #also includes the current timestep

        # Get the target datapoint ID
        target_index = index + self.prediction_len
        target_datapoint_id = self.dataset.loc[target_index, 'id']

        # Retrieve sequence agents → all agents present within sequence timesteps
        sequence_agent_ids = set()
        agent_frequency = Counter()  # Counter for the frequency of agent IDs
        for sequence_datapoint_id in sequence_datapoint_ids:
            # Extract all agent IDs from the sequence datapoints
            sequence_agents = self.input_information.get(sequence_datapoint_id, {}).keys()
            # Update the set with agent IDs seen in the sequence
            sequence_agent_ids.update(sequence_agents)
            # Update the frequency counter for each agent ID
            agent_frequency.update(sequence_agents)

        # Retrieve target agents → all agents present in the future timestep
        target_agents = self.target_information.get(target_datapoint_id, {})
        target_agent_ids = list(target_agents.keys())

        # Union of all agents appearing in sequence or future timestep
        all_agent_ids = list(sequence_agent_ids | set(target_agent_ids))

        # Ensure the number of agents does not exceed the defined maximum
        assert len(all_agent_ids) <= self.max_agents

        #Binary classification logit
        # 4   Agent present in sequence but missing in target
        # 3   New agent appearing only in target
        # 2   Agent present in both sequence and target, but with insufficient history (not observed in enough timesteps = min_timespes_seen)
        # 1   Agent present in both sequence and target, with sufficient history → Predict
        # 0   Padding

        # Prepare the target tensor
        # If target_agents is empty, create a tensor of zeros
        # (Shape: max_agents, num_features)
        target_tensor = torch.zeros(self.max_agents, 13) #9, 4
        for i, agent_id in enumerate(all_agent_ids):
            if agent_id in target_agents:
                # Get how often the agent appeared in the sequence
                seen_agent_counts = agent_frequency.get(agent_id, 0)
                # Assign agent logit based on its presence and visibility history
                logit = (
                    1 if agent_id in sequence_agent_ids and seen_agent_counts >= self.min_timesteps_seen  # Persistent agent with sufficient history
                    else 2 if agent_id in sequence_agent_ids and seen_agent_counts < self.min_timesteps_seen  # Persistent agent with insufficient history
                    else 3  # New agent appearing only in the target
                )
                # Assign true future values
                target_tensor[i] = target_agents[agent_id]
                target_tensor[i, 0] = logit
            else:
                target_tensor[i] = torch.tensor([4.0] + [0.0] * 12)   # Agent no longer exists in the target

        # Prepare the input tensor
        input_tensor_list = []
        # For each agent, gather its features across the full sequence length
        for agent_id in all_agent_ids:
            agent_sequence = []
            # Iterate through all timesteps in the sequence
            for sequence_datapoint_id in sequence_datapoint_ids:
                # Get all agent observations at the current timestep
                agent_info = self.input_information.get(sequence_datapoint_id, {})
                seen_agent_counts = agent_frequency.get(agent_id, 0)
                # Assign agent logit based on its presence and visibility history
                logit = (
                    1 if agent_id in target_agent_ids and agent_id in sequence_agent_ids and seen_agent_counts >= self.min_timesteps_seen  # Predictable agent
                    else 2 if agent_id in target_agent_ids and agent_id in sequence_agent_ids and seen_agent_counts < self.min_timesteps_seen  # Insufficient history
                    else 3 if agent_id in target_agent_ids and agent_id not in sequence_agent_ids  # Newly appearing agent
                    else 4)  # Agent is not part of the target

                if agent_id in agent_info:
                    # Agent is visible at this timestep – use actual features
                    agent_tensor = agent_info[agent_id]
                    agent_tensor[0] = logit
                    agent_sequence.append(agent_tensor)
                else:
                    # Agent not observed at this timestep but exists in the future (sequence_len or target) – insert placeholder with logit only
                    agent_sequence.append(torch.tensor([logit] + [0.0] * 12))

            # Stack all timestep tensors into one tensor per agent: shape (sequence_len, num_features)
            agent_sequence = torch.stack(agent_sequence)
            input_tensor_list.append(agent_sequence)

        # If no agents are present, return a fully zero-initialized tensor
        if len(input_tensor_list) == 0:
            input_tensor = torch.zeros(self.sequence_len, self.max_agents, 13)
        else:
            # Combine all agent tensors into a single tensor: shape (num_agents, sequence_len, num_features)
            input_tensor_stacked = torch.stack(input_tensor_list)  # Stack the input tensors
            input_tensor_stacked = rearrange(input_tensor_stacked, 'v s c -> s v c')  # Rearrange
            # Initialize the full input tensor with zeros
            input_tensor = torch.zeros(self.sequence_len, self.max_agents, 13)
            # Fill in the available agent data
            input_tensor[:, :len(all_agent_ids), :] = input_tensor_stacked

        # During training, randomly shorten the input sequence to simulate variable history lengths,
        # making the model robust for autoregressive prediction with increasing context over multiple timesteps
        # Pad it back to the original sequence length to maintain consistent tensor shape
        if self.max_prediction != None:
            # Randomly shorten the input sequence to simulate a limited observation window
            random_length = random.randint(self.sequence_len-self.max_prediction, self.sequence_len-1)
            # If the current input sequence is longer than the random length, keep only the last part
            if input_tensor.shape[0] > random_length:
                input_tensor = input_tensor[-random_length:] # Trim the sequence to retain only the last `random_length` timesteps

            # Pad the beginning of the sequence with zeros if it's shorter than the required input length
            if input_tensor.shape[0] < self.sequence_len:
                padding_size = self.sequence_len - input_tensor.shape[0]
                padding = torch.zeros(padding_size, input_tensor.shape[1], input_tensor.shape[2],
                                      device=input_tensor.device)
                input_tensor = torch.cat([padding, input_tensor], dim=0)  # Prepend padding to align with original length

        return input_tensor, target_tensor, index

    def _get_allowed_indexes(self) -> None:
        """
          Identifies all valid sample indexes in the dataset that can be used for training or evaluation, i.e., for which
          a complete input sequence and prediction horizon are available.
        """

        # Initialize allowed indexes
        self.allowed_indexes = []

        # Group the dataset by 'dataset_name' and 'loop'
        grouped = self.dataset.groupby(['dataset_name', 'loop'])

        for (dataset_name, loop), group in tqdm(grouped, desc='Finding allowed indexes'):
            # Sort the group by timestep to maintain temporal order
            group = group.sort_values('timestep').reset_index()
            indices = group['index'].tolist()

            # Generate allowed indexes within each combination of 'dataset_name' and 'loop'
            if len(indices) >= self.sequence_len + self.prediction_len - 1:
                # Start from index (sequence_len - 1) to ensure enough history is available
                for i in range(self.sequence_len - 1, len(indices) - self.prediction_len):
                    self.allowed_indexes.append(indices[i])


    def _create_inputs(self) -> None:
        """
        Creates input feature vectors per agent. Features are normalized:
            - Position is normalized relative to the centerpoint and radius.
            - Speed and acceleration is normalized via Z-Score (standardization).
            - Orientation is normalized via sinus and cosinus of the heading angle.
            - Agent type is represented by a numerical value ranging from 0 to 7.
            - Route information/exit is represented by a four-dimensional one-hot-encoded vector.
        """

        # Prepare input tensors for all data points
        self.input_information = {}

        for data in tqdm(self.dataset.itertuples(), total=len(self.dataset), desc="Preparing input tensors"):
            processed_agent_information = {}

            for agent_id, agent_data in data.agent_information.items():
                # One-Hot Encoding for the exit
                one_hot_exit = [0] * 4  # Initialize a list with 4 zeros according to the roundabout's topology
                # Retrieve the encoded exit
                one_hot_exit[agent_data['exit']] = 1

                normalized_features = [
                    -1,  # Assuming this is a fixed label or value (placeholder for binary classification logit)
                    (agent_data['position'][0] - self.center_point[0]) / self.radius,
                    (agent_data['position'][1] - self.center_point[1]) / self.radius,
                    (agent_data['speed'] - self.mean_speed) / self.std_speed,
                    (agent_data['acceleration_lat'] - self.mean_acceleration_lat) / self.std_acceleration_lat,
                    (agent_data['acceleration_tan'] - self.mean_acceleration_tan) / self.std_acceleration_tan,
                    math.sin(agent_data['angle']),
                    math.cos(agent_data['angle']),
                    self.type_encoding.get(agent_data['type'], -1)
                ]

                normalized_features.extend(one_hot_exit)

                # Calculate the distance and filter agents within a unit radius
                distance = math.sqrt(normalized_features[1] ** 2 + normalized_features[2] ** 2)
                if distance <= 1:
                    processed_agent_information[agent_id] = torch.tensor(normalized_features)

            # Store processed agent information for this data point
            self.input_information[data.id] = processed_agent_information

    def _create_targets(self) -> None:
        """
        Creates target feature vectors per agent. Features are normalized:
            - Position is normalized relative to the centerpoint and radius.
            - Speed and acceleration is normalized via Z-Score (standardization).
            - Orientation is normalized via sinus and cosinus of the heading angle.
            - Agent type is represented by a numerical value ranging from 0 to 7.
            - Route information/exit is represented by a four-dimensional one-hot-encoded vector.
        """

        # Prepare input tensors for all data points
        self.target_information = {}

        for data in tqdm(self.dataset.itertuples(), total=len(self.dataset), desc="Preparing target tensors"):
            processed_agent_information = {}

            for agent_id, agent_data in data.agent_information.items():
                # One-Hot Encoding for the exit
                one_hot_exit = [0] * 4  # Initialize a list with 4 zeros according to the roundabout's topology
                # Retrieve the encoded exit
                one_hot_exit[agent_data['exit']] = 1

                normalized_features = [
                    -1,  # Assuming this is a fixed label or value (placeholder for binary classification logit)
                    (agent_data['position'][0] - self.center_point[0]) / self.radius,
                    (agent_data['position'][1] - self.center_point[1]) / self.radius,
                    (agent_data['speed'] - self.mean_speed) / self.std_speed,
                    (agent_data['acceleration_lat'] - self.mean_acceleration_lat) / self.std_acceleration_lat,
                    (agent_data['acceleration_tan'] - self.mean_acceleration_tan) / self.std_acceleration_tan,
                    math.sin(agent_data['angle']),
                    math.cos(agent_data['angle']),
                    self.type_encoding.get(agent_data['type'], -1)
                ]

                normalized_features.extend(one_hot_exit)

                # Calculate the distance and filter agents within a unit radius
                distance = math.sqrt(normalized_features[1] ** 2 + normalized_features[2] ** 2)
                if distance <= 1:
                    processed_agent_information[agent_id] = torch.tensor(normalized_features)

            # Store processed agent information for this data point
            self.target_information[data.id] = processed_agent_information

            # Ensure the number of agents does not exceed max_agents
            assert len(processed_agent_information) <= self.max_agents
            if len(processed_agent_information) > self.max_agents_counter:
                self.max_agents_counter = len(processed_agent_information)

    def _split_dataset(self,
            dataset: pd.DataFrame,
            splits: Union[Tuple[Union[int, float, str], Union[int, float, str]], List[
                Tuple[Union[int, float, str], Union[int, float, str]]]]
    ) -> pd.DataFrame:
        """
        Splits the dataset based on the provided index or ratio ranges.

        Args:
            dataset (pd.DataFrame): The full dataset to be split.
            splits (tuple or list of tuples): Start and end indices or fractions. Examples: (0.0, 0.8), [(':', 0.5), (0.8, ':')]

        Returns:
            pd.DataFrame: The concatenated subset(s) of the dataset based on the split ranges.
        """

        if splits is not None:
            # If splits is a list of split ranges (e.g., multiple intervals)
            if isinstance(splits, list):
                datasets = []
                for split in splits:
                    # Determine the start index:
                    # If ':' is given, start from beginning (0),
                    # If a float, interpret it as a proportion of dataset length,
                    # Otherwise, use the integer directly.
                    start = 0 if split[0] == ':' else (
                        int(split[0] * len(dataset)) if isinstance(split[0], float) else split[0]
                    )
                    # Determine the end index
                    end = len(dataset) if split[1] == ':' else (
                        int(split[1] * len(dataset)) if isinstance(split[1], float) else split[1]
                    )
                    # Append the selected slice of the dataset
                    datasets.append(dataset.iloc[start:end])
                # Concatenate all slices into one dataset and reset indices
                return pd.concat(datasets, ignore_index=True)
            else:
                # Single split range (tuple or list of two elements)
                # Determine the start index
                start = 0 if splits[0] == ':' else (
                    int(splits[0] * len(dataset)) if isinstance(splits[0], float) else splits[0]
                )
                # Determine the end index
                end = len(dataset) if splits[1] == ':' else (
                    int(splits[1] * len(dataset)) if isinstance(splits[1], float) else splits[1]
                )
                # Return the sliced dataset
                dataset = dataset.iloc[start:end]
                return dataset
        else:
            # If no splits provided, return the full dataset unchanged
            return dataset

    def _compute_mean_std(self) -> None:
        """
        Computes the mean and standard deviation of speed, lateral acceleration,
        and tangential acceleration across all agents in the dataset.
        Will be used for feature normalization (Z-Score).
        """

        speeds = []
        accelerations_lat = []
        accelerations_tan = []

        # Iterate over all rows in the dataset
        for _, row in self.dataset.iterrows():
            if 'agent_information' in row:
                # Iterate over all agents in the row
                for agent_id, agent_data in row['agent_information'].items():
                    speeds.append(agent_data['speed'])
                    accelerations_lat.append(agent_data['acceleration_lat'])
                    accelerations_tan.append(agent_data['acceleration_tan'])

        # Convert lists to tensors and compute statistics if data exists
        if speeds and accelerations_lat and accelerations_tan:
            speeds = torch.tensor(speeds, dtype=torch.float32)
            accelerations_lat = torch.tensor(accelerations_lat, dtype=torch.float32)
            accelerations_tan = torch.tensor(accelerations_tan, dtype=torch.float32)

            self.mean_speed = speeds.mean().item()
            self.std_speed = speeds.std().item()
            self.mean_acceleration_lat = accelerations_lat.mean().item()
            self.std_acceleration_lat = accelerations_lat.std().item()
            self.mean_acceleration_tan = accelerations_tan.mean().item()
            self.std_acceleration_tan = accelerations_tan.std().item()

        else:
            # Use fallback values if no data is available
            self.mean_speed, self.std_speed = 0.0, 1.0
            self.mean_acceleration_lat, self.std_acceleration_lat = 0.0, 1.0
            self.mean_acceleration_tan, self.std_acceleration_tan = 0.0, 1.0


if __name__ == "__main__":
    raise NotImplementedError("This script is not meant to be executed")
