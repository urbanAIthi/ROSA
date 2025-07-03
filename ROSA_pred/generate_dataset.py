import sqlite3
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from typing import Tuple, List


class DatasetPreprocessor:
    """
    Preprocesses roundabout trajectory data, especially for assigning exit directions
    based on object positions and scenario-specific logic.

    Args:
        roundabout_scenario (str): Identifier for the roundabout scenario from the config (e.g., 'rdb1', 'rdb6').
        original_base (str): Path to the original dataset.
        rosa_base (str): Path to the ROSA-formatted dataset.
        shapefile_path (str): Path to the shapefile containing map information of the selected roundabout scenario.
        center_point (Tuple[float, float]): Coordinates of the roundabout center for angle calculations.
    """


    def __init__(self, roundabout_scenario: str, original_base: str, rosa_base: str, shapefile_path: str, centerpoint: Tuple[float, float]) -> None:
        self.roundabout_scenario = roundabout_scenario
        self.original_base = original_base
        self.rosa_base = rosa_base
        self.shapefile_path = shapefile_path
        self.center_point = centerpoint
        self.gdf = gpd.read_file(self.shapefile_path)

    def determine_exit(self, x: float, y: float, agent_class: str) -> Tuple[int, str]:
        """
        Determines the exit direction of a vehicle based on its position and the roundabout scenario.

        Args:
            x (float): x-coordinate of the agent.
            y (float): y-coordinate of the agent.
            agent_class (str): Type of agent (e.g., 'Car', 'Pedestrian').

        Returns:
            Tuple[int, str]: A tuple containing:
                - exit_id (int): The assigned exit (0, 1, 2, or -1 if undefined or not applicable).
                - color (str): A color used for visualization corresponding to the exit.
        """

        # Non-motorized agents don't get an exit assigned
        if agent_class in ['Pedestrian', 'Bicycle']:
            return -1, 'gray'

        # Ignore agents' final positions too close to the center (within the roundabout)
        distance_from_center = np.sqrt((x - self.center_point[0]) ** 2 + (y - self.center_point[1]) ** 2)
        if distance_from_center <= 18:
            return -1, 'gray'

        # Calculate angle relative to the center of the roundabout, scenario-specific
        if self.roundabout_scenario == 'rdb1':
            angle = np.degrees(np.arctan2(y - self.center_point[1], x - self.center_point[0])) - 10
            if angle < -180:
                angle += 360
            elif angle > 180:
                angle -= 360
        elif self.roundabout_scenario == 'rdb6':
            angle = np.degrees(np.arctan2(y - self.center_point[1], x - self.center_point[0]))
        else:
            raise NotImplementedError("Roundabout scenario not defined.")

        # Assign exit ID and color based on angle sector
        if -45 <= angle < 45:
            return 1, 'blue'
        elif 45 <= angle < 135:
            return 0, 'green'
        elif angle >= 135 or angle < -135:
            return 2, 'red'
        elif -135 <= angle < -45:
            return -1, 'yellow'
        else:
            return -1, 'gray'

    def create_dataset_with_exit(self) -> None:
        """
        Creates a new dataset by copying tables from the original SQLite database,
        adding an 'Exit' column, and populating it based on the agents' last positions.
        """

        # Connect to the original database (read source data)
        conn_existing = sqlite3.connect(self.original_base)
        cursor_existing = conn_existing.cursor()

        # Connect to the new database where modified tables with 'Exit' column will be created
        conn_new = sqlite3.connect(self.rosa_base)
        cursor_new = conn_new.cursor()

        # Retrieve all table names from the original database
        cursor_existing.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor_existing.fetchall()

        # Iterate over all tables in the original database
        for table in tables:
            table_name = table[0]

            # Get column info to replicate the table structure
            cursor_existing.execute(f"PRAGMA table_info('{table_name}');")
            columns = cursor_existing.fetchall()
            # Build the column definitions string, adding an extra 'Exit' INTEGER column
            column_definitions = ", ".join([f"{col[1]} {col[2]}" for col in columns])
            cursor_new.execute(f"CREATE TABLE IF NOT EXISTS '{table_name}' ({column_definitions}, Exit INTEGER);")

            # Fetch all rows from the original table
            cursor_existing.execute(f"SELECT * FROM '{table_name}';")
            rows = cursor_existing.fetchall()
            # Prepare placeholders for each column, adding one for the new 'Exit' column
            placeholders = ", ".join(["?" for _ in columns])
            # Insert all rows into the new table, initializing 'Exit' with NULL
            cursor_new.executemany(f"INSERT INTO '{table_name}' VALUES ({placeholders}, NULL);", rows)

            # Query distinct agent IDs and their classes in the new table
            cursor_new.execute(f"SELECT DISTINCT OBJID, CLASS FROM '{table_name}';")
            agents = cursor_new.fetchall()

            # For each agent, determine their exit based on their last known position
            for agent_id, agent_class in agents:
                # Get the latest UTM coordinates for the agent by timestamp descending
                cursor_new.execute(
                    f"SELECT UTM_X, UTM_Y FROM '{table_name}' WHERE OBJID = ? ORDER BY TIMESTAMP DESC LIMIT 1;",
                    (agent_id,))
                last_pos = cursor_new.fetchone()
                if last_pos:
                    # Use the determine_exit method to calculate the exit ID
                    exit_value, _ = self.determine_exit(last_pos[0], last_pos[1], agent_class)
                    # Update the 'Exit' column for all rows of this agent
                    cursor_new.execute(f"UPDATE '{table_name}' SET Exit = ? WHERE OBJID = ?;", (exit_value, agent_id))

            # Commit changes for this table before moving on
            conn_new.commit()

        # Close both database connections
        conn_existing.close()
        conn_new.close()
        print("ROSA base with exit information successfully generated.")

    def visualize_trajectories(self) -> None:
        """
        Visualizes exit angle sectors and classfied agent trajectories from the ROSA database on a map of the roundabout.
        """

        # Connect to the ROSA database
        conn_new = sqlite3.connect(self.rosa_base)
        cursor_new = conn_new.cursor()

        # Get the first table name from the database (assumes at least one table exists)
        cursor_new.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;")
        first_table = cursor_new.fetchone()
        if not first_table:
            print("No tables found.")
            return

        table_name = first_table[0]
        print(f"Visualize table: {table_name}")

        # Query up to 10 agents that have at least 5 position records (to visualize meaningful trajectories)
        query_example_agent = f"SELECT OBJID, CLASS FROM '{table_name}' GROUP BY OBJID HAVING COUNT(*) >= 5 LIMIT 10"
        cursor_new.execute(query_example_agent)
        agents = cursor_new.fetchall()

        # Setup plot: base roundabout map and buffer circle around center point
        fig, ax = plt.subplots(figsize=(8, 8))
        self.gdf.plot(ax=ax, color='lightblue', edgecolor='black')
        roundabout_center = Point(self.center_point[0], self.center_point[1])
        gpd.GeoSeries([roundabout_center.buffer(20)]).plot(ax=ax, color='none', edgecolor='black', linestyle='--')

        max_radius = 100  # radius length for sector visualization
        shift_angle = 20  # angle offset for scenario rdb1

        # Draw colored sectors representing exits around the roundabout center
        for i, color in enumerate(['blue', 'green', 'red', 'yellow']):

            if self.roundabout_scenario == 'rdb1':
                angle_start, angle_end = (i * 90 - 45 + shift_angle) % 360, ((i + 1) * 90 - 45 + shift_angle) % 360
                if angle_start < 0:
                    angle_start += 360
                if angle_end < 0:
                    angle_end += 360
            elif self.roundabout_scenario == 'rdb6':
                angle_start, angle_end = i * 90 - 45, (i + 1) * 90 - 45
            else:
                raise NotImplementedError("Roundabout scenario not defined.")

            # Create polygon sector for visualization
            sector_polygon = Polygon([
                (self.center_point[0], self.center_point[1]),
                (self.center_point[0] + np.cos(np.radians(angle_start)) * max_radius,
                 self.center_point[1] + np.sin(np.radians(angle_start)) * max_radius),
                (self.center_point[0] + np.cos(np.radians(angle_end)) * max_radius,
                 self.center_point[1] + np.sin(np.radians(angle_end)) * max_radius)
            ])
            gpd.GeoSeries([sector_polygon]).plot(ax=ax, color=color, alpha=0.3)

        # Plot each agent's trajectory points, colored by their computed exit sector
        for agent_id, agent_class in agents:
            query_trajectory = f"SELECT UTM_X, UTM_Y FROM '{table_name}' WHERE OBJID = ? ORDER BY TIMESTAMP ASC"
            cursor_new.execute(query_trajectory, (agent_id,))
            trajectory = cursor_new.fetchall()

            if len(trajectory) < 2:
                print(f"Agent {agent_id} has too many positions.")
                continue

            last_x, last_y = trajectory[-1]
            _, sector_color = self.determine_exit(last_x, last_y, agent_class)
            traj_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in trajectory])
            traj_gdf.plot(ax=ax, color=sector_color, marker="o", markersize=5)

        # Set plot labels and title
        plt.title("Trajectories with exit")
        plt.xlabel("UTM X")
        plt.ylabel("UTM Y")
        plt.show()
        conn_new.close()


class ROSADatasetGenerator:
    """
    Generates the ROSA dataset for multi-agent trajectory prediction using a Transformer-based architecture.

    Args:
        rosa_base (str): Path to the ROSA SQLite database file.
        shapefile_path (str): Path to the shapefile describing the roundabout geometry.
        centerpoint (Tuple[float, float]): Coordinates (x, y) representing the roundabout center.
    """

    def __init__(self, rosa_base: str, shapefile_path: str, centerpoint: Tuple[float, float]) -> None:
        self.rosa_base = rosa_base
        self.shapefile_path = shapefile_path
        self.center_point = centerpoint
        self.tables = self.get_tables()


    def get_tables(self) -> List[str]:
        """
        Retrieves all table names from the ROSA SQLite database.

        Returns:
            tables (List[str]): List of table names found in the
        """

        conn = sqlite3.connect(self.rosa_base)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables

    def load_table(self, table: str) -> pd.DataFrame:
        """
        Loads all data from a specified table into a pandas DataFrame.

        Args:
            table (str): The name of the table to load.

        Returns:
            df (pd.DataFrame): DataFrame containing all rows from the table with an added 'loop' column indicating the table name.
        """

        conn = sqlite3.connect(self.rosa_base)
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        df["loop"] = table # Add column to track the source table
        conn.close()
        return df

    def process_data(self, table: str, df: pd.DataFrame, full_seconds_only: bool = False) -> pd.DataFrame:
        """
        Processes raw data from a table and create a structured dataset suitable for trajectory analysis.

        Args:
            table (str): Table name to identify the data source.
            df (pd.DataFrame): Input DataFrame containing raw data with TIMESTAMP and agent info.
            full_seconds_only (bool): If True, only the closest data points to full seconds are kept; otherwise, all timestamps are used.

        Returns:
            dataset (pd.DataFrame): Processed dataset where each row corresponds to a timestep,
                          containing agent information in a dictionary keyed by agent IDs.
        """

        dataset = pd.DataFrame(columns=["id", "loop", "timestep", "agent_information"])

        if full_seconds_only:
            # Generate array of full seconds between min and max timestamps
            # Get the unique full second timestamps
            full_seconds = np.arange(int(df['TIMESTAMP'].min()), int(df['TIMESTAMP'].max()) + 1)
            timestamps_to_process = []

            # For each full second, find the closest timestamp in the data >= that second
            for second in full_seconds:
                # Find the closest timestamp for the current second
                closest_timestamp = df[df["TIMESTAMP"] >= second]["TIMESTAMP"].min()
                if not np.isnan(closest_timestamp):
                    timestamps_to_process.append(closest_timestamp)
        else:
            # Use all unique timestamps from the dataset
            timestamps_to_process = df["TIMESTAMP"].unique()

        for timestamp in timestamps_to_process:
            # Group all agents recorded at the current timestamp
            group = df[df["TIMESTAMP"] == timestamp]

            # Create a dictionary of agent info keyed by OBJID
            agent_info = {
                row["OBJID"]: {
                    "position": (row["UTM_X"], row["UTM_Y"]),
                    "angle": row["UTM_ANGLE"],
                    "speed": row["V"],
                    "acceleration": row["ACC"],
                    "acceleration_lat": row["ACC_LAT"],
                    "acceleration_tan": row["ACC_TAN"],
                    "type": row["CLASS"],
                    "width": row["WIDTH"],
                    "length": row["LENGTH"],
                    "trailer_id": row["TRAILER_ID"],
                    "exit": row["Exit"]
                }
                for _, row in group.iterrows()
            }

            # Warn if no agents are found at this timestamp
            if not agent_info:
                print(f"No agent information for timestamp {timestamp:.6f} in table {table}")

            # Construct a single-row DataFrame for this timestep
            datapoint = pd.DataFrame({
                "id": f"{table.split('_')[1]}_{timestamp:.6f}",
                "loop": table,
                "timestep": timestamp,
                "agent_information": [agent_info],
            })

            # Append the datapoint to the overall dataset
            dataset = pd.concat([dataset, datapoint], ignore_index=True)

        return dataset

    def save_dataset(self, dataset: pd.DataFrame, output_file: str) -> None:
        """
        Saves the processed dataset as a pickle file.

        Args:
            dataset (pd.DataFrame): The processed ROSA dataset to save.
            output_file (str): File path to save the pickle file.
        """

        dataset.to_pickle(output_file)
        print(f"ROSA dataset successfully saved to {output_file}!")


if __name__ == "__main__":
    raise NotImplementedError("This script is not intended to be run directly.")
