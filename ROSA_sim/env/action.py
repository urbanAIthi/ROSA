import configparser
import logging
import os
import pandas as pd

class ActionDefinition():
    """
    Defines the behavior and action execution for a simulated vehicle agent
    based on configuration parameters and prediction data.
    This class executes vehicle actions and traffic light control during simulation. The traffic lights represent the predicted occupany status at the roundabout, red for occupied and green for free.

    Args:
        config (configparser.ConfigParser): Configuration parser object with simulation settings.
        logger (logging.Logger): Logger instance for tracking experiment progress and issues.
        ad (int): Offset time to the base time for the ego vehicle to start in the simulation (only relevant in evaluation mode).
    """

    def __init__(self, config: configparser.ConfigParser, logger: logging.Logger, ad: int) -> None:

        self.config = config
        self.logger = logger
        self.ad = ad

        self.base = self.config.getint('Simulation', 'base_starttime')
        self.agent = self.config.get('General', 'agent')
        self.max_speed = self.config.getfloat('General', 'max_speed')
        self.min_speed = self.config.getfloat('General', 'min_speed')
        self.max_deceleration = self.config.getfloat('General', 'max_deceleration')
        self.max_acceleration = self.config.getfloat('General', 'max_acceleration')

        prediction = pd.ExcelFile(os.path.join(config.get('Prediction', 'model_path'), config.get('Prediction', 'occupancy_file')))
        self.vru_phases = pd.read_excel(prediction, sheet_name="Prediction_5", usecols=["state_target_vru"])
        self.veh_phases = pd.read_excel(prediction, sheet_name="Prediction_5", usecols=["state_target_veh"])
        self.vru_phases.index = self.vru_phases.index.astype(int)
        self.veh_phases.index = self.veh_phases.index.astype(int)

        # Choose SUMO interface: libsumo or traci
        if self.config.getboolean('Simulation', 'libsumo'):
            import libsumo as traci
        else:
            import traci
        self.traci = traci


    def set_action(self, action: float, vehicle_id: str, pre_eval) -> None:
        """
        Sets the action of the ego vehicle and controls the traffic light states based on predicted occupancy.

        Args:
            action (float): Action to be executed.
            vehicle_id (str): Identifier of the vehicle to control.
            pre_eval (bool): Flag indicating if the simulation is in the pre-evaluation phase without ROSA.
        """

        # Define traffic light IDs for vehicle and VRU occupancy
        tl_id_veh = '7166206298'
        tl_id_vru = '7166186980'
        # Set traffic lights to manual mode
        self.traci.trafficlight.setProgram(tl_id_vru, "0")
        self.traci.trafficlight.setProgram(tl_id_veh, "0")

        # Compute the prediction index based on current simulation time and base offset
        index = int(self.traci.simulation.getTime() - self.base - 4)  # offset due to 5s ahead prediction

        # Set traffic light states for roundabout entries based on predicted occupancy by vehicles
        if index in self.veh_phases.index:
            state = self.veh_phases.loc[index, "state_target_veh"]
            self.traci.trafficlight.setRedYellowGreenState(tl_id_veh, 'GG' if state == 'g' else 'rG')
        else:
            self.traci.trafficlight.setRedYellowGreenState(tl_id_veh, 'GG')  # default to green

        # Set traffic light states for crosswalks based on predicted occupancy by VRUs
        if index in self.vru_phases.index:
            state = self.vru_phases.loc[index, "state_target_vru"]
            self.traci.trafficlight.setRedYellowGreenState(tl_id_vru, 'GGG' if state == 'g' else 'GrG')
        else:
            self.traci.trafficlight.setRedYellowGreenState(tl_id_vru, 'GGG')  # default to green

        # If in pre-evaluation phase, only set default vehicle dynamics
        if pre_eval:
            self.set_pre_eval(vehicle_id)
            return

        self._set_speed(action, vehicle_id)

    def set_pre_eval(self, vehicle_id: str) -> None:
        """
        Sets the vehicly dynamic parameters during the pre evaluation phase to be comparable in the evaluation

         Args:
            vehicle_id (str): Identifier of the vehicle to control.
        """

        # Set the max and min deceleration of the ego vehicle according to the configuration
        self.traci.vehicle.setDecel(vehicle_id, self.max_deceleration)
        self.traci.vehicle.setAccel(vehicle_id, self.max_acceleration)
        self.traci.vehicle.setMaxSpeed(vehicle_id, self.max_speed)
        self.traci.vehicle.setEmergencyDecel(vehicle_id, 0.8)
        self.traci.vehicle.setTau(vehicle_id, 0.1)

    def _set_speed(self, action: float, vehicle_id: str) -> None:
        """
        This function sets the speed of the ego vehicle according to the action definition.

         Args:
            action (float): Action to be executed.
            vehicle_id (str): Identifier of the vehicle to control.
        """

        self.logger.info(f'++++++++++++++++++++++ action {action}')

        # Set the max and min deceleration of the ego vehicle according to the configuration
        self.traci.vehicle.setAccel(vehicle_id, self.max_acceleration)
        self.traci.vehicle.setMaxSpeed(vehicle_id, self.max_speed)
        self.traci.vehicle.setEmergencyDecel(vehicle_id, 0.8)
        self.traci.vehicle.setTau(vehicle_id, 0.1)

        # Execute the ROSA action
        self.traci.vehicle.setDecel(vehicle_id, action[1])
        if self.traci.vehicle.getSpeed(vehicle_id) < self.min_speed:
            self.traci.vehicle.setSpeed(vehicle_id, -1)
        else:
            self.traci.vehicle.setSpeed(vehicle_id, action[0])


if __name__ == "__main__":
    raise NotImplementedError("This script is not intended to be run directly.")