import pandas as pd
import numpy as np
import wandb
import logging
import os
import imageio
import plotly.graph_objs as go
import configparser
from typing import Tuple, Dict, List

from env.rosa import ROSA_agent
from env.environment import SumoEnv


class Evaluator:
    """
    Evaluates vehicle performance using the SUMO simulation, such as speed, acceleration, emissions, and energy consumption. Optionally, screenshots and videos can
    be taken if the GUI is enabled.

    Args:
        config (configparser.ConfigParser): Configuration parser object with simulation settings.
        logger (logging.Logger): Logger instance for tracking experiment progress and issues.
        path (str): Path to store visualizations of perfomance and screenshots/video if GUI is enabled.
        gui (bool): If True, enables screenshot capture at each simulation step.
    """

    def __init__(self, config: configparser.ConfigParser, logger: logging.Logger, path: str, gui: bool = False) -> None:

        self.config = config
        self.logger = logger
        self.path = path
        self.gui = gui

        self.speed = {}
        self.acceleration = {}
        self.distance = {}
        self.co2 = {}
        self.fuel = {}
        self.energy = {}

        self.vehicle_id = 'ego'

        # Choose SUMO interface: libsumo or traci
        if self.config.getboolean('Simulation', 'libsumo'):
            import libsumo as traci
        else:
            import traci
        self.traci = traci


    def get_info(self) -> None:
        """
        Retrieves current performance metrics for the ego vehicle at the current simulation time
        and stores them in the corresponding dictionaries.

        This includes speed, acceleration, total traveled distance, CO2 emission, fuel consumption, and energy usage.
        If the GUI is enabled, it also captures a screenshot.
        """

        # Skip if the ego vehicle is not currently in the simulation
        if not self.vehicle_id in self.traci.vehicle.getIDList():
            pass
        else:
            # Collect simulation data for the ego vehicle
            time = self.traci.simulation.getTime()
            self.speed[time] = self.traci.vehicle.getSpeed(self.vehicle_id)
            self.acceleration[time] = self.traci.vehicle.getAcceleration(self.vehicle_id)
            self.distance[time] = self.traci.vehicle.getDistance(self.vehicle_id)
            self.co2[time] = self.traci.vehicle.getCO2Emission(self.vehicle_id)
            self.fuel[time] = self.traci.vehicle.getFuelConsumption(self.vehicle_id)
            self.energy[time] = self.traci.vehicle.getElectricityConsumption(self.vehicle_id)

            # Optionally save screenshot if GUI mode is active
            if self.gui:
                self._save_screenshot(time)

    def evaluate_infos(self) -> Tuple[float, float, float, float, float, int]:
        """
        Evaluates and summarizes key performance and vehicle metrics from the current scenario.
        Generates plots, computes metrics, and saves the raw results in an Excel.

        Returns:
            time_on_site (float): Total duration of the current simulation scenario.
            co2_emission (float): Cumulative CO2 emissions over the simulation period.
            fuel_consumption (float): Total fuel consumption.
            waiting_time (float): Total time with speed below 1 m/s (i.e., vehicle is considered waiting).
            energy_consumption (float): Total energy consumed.
            num_stops (int): Number of complete stops (speed drops below 1 m/s from >= 1 m/s).
        """

        # Generate visualizations of various metrics over time
        self._create_linechart(self.distance, 'time', 'distance', 'distance_over_time')
        self._create_linechart(self.speed, 'Time [s]', 'Speed [m/s]', 'speed_over_time')
        self._create_linechart(self.acceleration, 'time', 'acceleration', 'acceleration_over_time')
        self._create_linechart(self.fuel, 'time', 'fuel', 'fuel_over_time')
        self._create_linechart(self.co2, 'time', 'co2', 'co2_over_time')
        self._create_linechart(self.energy, 'time', 'energy', 'energy_over_time')

        # Save video if GUI mode is enabled
        if self.gui:
            self._save_video()

        # Compute aggregate performance metrics
        time_on_site = list(self.speed.keys())[-1] - list(self.speed.keys())[0]
        co2_emission = np.sum(list(self.co2.values()))
        fuel_consumption = np.sum(list(self.fuel.values()))
        waiting_time = np.sum(np.array(list(self.speed.values())) < 1)
        energy_consumption = np.sum(list(self.energy.values()))

        # Compute number of stops (speed drops below 1 after being >= 1)
        speeds_array = np.array(list(self.speed.values()))
        stopped_mask = (speeds_array < 1) & (np.roll(speeds_array, 1) >= 1)
        num_stops = np.sum(stopped_mask)

        # Create a summary dictionary for the performance metrics
        performance_metrics = {
            'energy_consumption': energy_consumption,
            'fuel_consumption': fuel_consumption,
            'co2_emission': co2_emission,
            'waiting_time': waiting_time,
            'time_on_site': time_on_site,
            'number_stops': num_stops
        }

        df_performance_metrics = pd.DataFrame([performance_metrics])

        # Create a time-indexed dictionary of detailed vehicle metrics
        vehicle_metrics = {}
        for t in self.speed:
            vehicle_metrics[t] = {
                'distance': self.distance[t],
                'speed': self.speed[t],
                'acceleration': self.acceleration[t],
                'fuel': self.fuel[t],
                'co2': self.co2[t],
                'energy': self.energy[t]
            }

        df_vehicle_metrics = pd.DataFrame.from_dict(vehicle_metrics, orient='index')
        df_vehicle_metrics.index.name = 'time'

        # Write both metrics to an Excel file with two sheets
        excel_path = os.path.join(self.path, 'evaluation_results.xlsx')
        with pd.ExcelWriter(excel_path) as writer:
            df_performance_metrics.to_excel(writer, sheet_name='Performance Metrics', index=False)
            df_vehicle_metrics.to_excel(writer, sheet_name='Vehicle Metrics')

        return time_on_site, co2_emission, fuel_consumption, waiting_time, energy_consumption, num_stops

    def _create_linechart(self, d: Dict[float, float], x_label: str, y_label: str, title: str) -> None:
        """
        Creates and saves an interactive HTML line chart using Plotly.

        Args:
            d (Dict[float, float]): Dictionary with time as keys and metric values as values.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            title (str): Title of the chart and filename for saving the HTML.
        """

        # Create a trace for the line chart using the data dictionary
        trace = go.Scatter(
            x=list(d.keys()),
            y=list(d.values()),
            mode='lines'
        )

        # Define the chart layout including axis labels and background
        layout = go.Layout(title=title,
                           xaxis=dict(title=x_label),
                           yaxis=dict(title=y_label),
                           plot_bgcolor = '#ffffff',
                           showlegend=True,
                           )

        # Generate the figure and save it as an HTML file
        fig = go.Figure(data=[trace], layout=layout)
        fig.write_html(os.path.join(self.path, f'{title}.html'))

    def _save_screenshot(self, time) -> None:
        """
        Captures and saves a screenshot from the SUMO GUI.

        Args:
            time (float): The current simulation time used to label the screenshot file.
        """

        os.makedirs(os.path.join(self.path, 'screenshots'), exist_ok=True)
        # Focus the SUMO GUI camera on the vehicle and take a screenshot
        self.traci.gui.trackVehicle('View #0', self.vehicle_id)
        self.traci.gui.setZoom('View #0', 500)
        self.traci.gui.screenshot('View #0', os.path.join(self.path, 'screenshots', f'{int(time)}.png'))

    def _save_video(self) -> None:
        """
        Compiles saved screenshots into a replay video.
        """

        image_folder = os.path.join(self.path, 'screenshots')  # Folder containing screenshot PNGs
        video_name = os.path.join(self.path, 'replay.mp4')   # Output video file path

        # Collect all .png images from the folder
        images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
        images.sort()  # Ensure the images are ordered correctly by filename (e.g., 1.png, 2.png, ...)

        # Create and save the video if images are available
        if images:
            with imageio.get_writer(video_name, fps=30) as video:
                for image in images:
                    image_path = os.path.join(image_folder, image)
                    video.append_data(imageio.imread(image_path))

def evaluate(config: configparser.ConfigParser, logger: logging.Logger, path: str, sumo_path: str, pre_eval: bool = True,
             gui: bool = False) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[int]]:
    """
    Evaluates the performance of the ROSA agent across multiple test scenarios. Also allows for comparison with a baseline without ROSA.

    Args:
        config (configparser.ConfigParser): Configuration parser object with simulation settings.
        logger (logging.Logger): Logger instance for tracking experiment progress and issues.
        path (str): Path to the root directory of the evaluation run.
        sumo_path (str): Path to the SUMO configuration files.
        load (bool, optional): If True, loads the trained agent to make predictions. If False, runs baseline with full speed. Defaults to True.
        gui (bool): If True, the SUMO GUI is displayed. Defaults to False.

    Returns:
        time_on_sites (List[float]): List of time on site values.
        co2_emissions (List[float]): List of CO2 emission values.
        fuel_consumptions (List[float]): List of fuel consumption values.
        waiting_times (List[float]): List of waiting times.
        energy_consumptions (List[float]): List of energy consumption values.
        number_stops (List[int]): List of total stop counts.
    """

    # Initialize the storage lists for the different metrics to keep track of several test scenarios
    time_on_sites = list()
    co2_emissions = list()
    fuel_consumptions = list()
    waiting_times = list()
    energy_consumptions = list()
    number_stops = list()

    # Evaluate the performance for the different test scenarios
    for ad in eval(config.get('Simulation', 'evaluations')):
        scenario_path = os.path.join(path, str(ad))
        os.mkdir(scenario_path)

        # Initialize evaluator and environment for the current scenario
        evaluator = Evaluator(config, logger, scenario_path, gui=gui)
        env = SumoEnv(config, logger, sumo_path, evaluate=True, evaluator=evaluator, gui=gui, ad=ad)

        # Initialize agent based on configuration (currently supports 'classic')
        if config.get('General', 'agent') == 'classic':
            agent = ROSA_agent(config, logger, ad)
            pass

            # Reset environment before starting simulation loop
            env.reset()

            while True:

                # Baseline without ROSA: vehicle drives at full speed
                if pre_eval:
                    action = config.getfloat('General', 'max_speed') # To test normal baseline

                # Predict the action of the ROSA agent
                else:
                    action = agent.predict()

                # Apply the action and get termination signal
                dones = env.step(action, pre_eval=pre_eval)

                # Exit loop if episode is finished
                if dones:
                    break

            # Retrieve and store evaluation metrics for current scenario
            time_on_site, co2_emission, fuel_consumption, waiting_time, energy_consumption, num_stops = evaluator.evaluate_infos()
            time_on_sites.append(time_on_site)
            co2_emissions.append(co2_emission)
            fuel_consumptions.append(fuel_consumption)
            waiting_times.append(waiting_time)
            energy_consumptions.append(energy_consumption)
            number_stops.append(num_stops)

    # Log average metrics to Weights & Biases
    wandb.log({
        "time_on_site": np.mean(time_on_sites),
        "co2_emission": np.mean(co2_emissions),
        "fuel_consumption": np.mean(fuel_consumptions),
        "waiting_time": np.mean(waiting_times),
        "energy_consumption": np.mean(energy_consumptions),
        "number_stops": sum(number_stops)
    })

    return time_on_sites, co2_emissions, fuel_consumptions, waiting_times, energy_consumptions, number_stops


if __name__ == "__main__":
    raise NotImplementedError("This script is not intended to be run directly.")
