import os
import random
import configparser
import logging
import numpy as np
import xml.etree.ElementTree as ET
from gym import Env
from gym.spaces import Box

from env.action import ActionDefinition
from env.emissions import bev, ice
from utils.helpers import configure_sumo


class SumoEnv(Env):
    """
    Initializes the SUMO simulation environment.
    Inherits from gym.Env and implements required env interface methods.

    Args:
        config (configparser.ConfigParser): Configuration parser object with simulation settings.
        logger (logging.Logger): Logger instance for tracking experiment progress and issues.
        sumo_path (str): Path to the SUMO configuration files.
        evaluate (bool): If True, the environment is used in evaluation mode.
        evaluator: Evaluation object used to log performance (only relevant in evaluation mode).
        gui (bool): If True, the SUMO GUI is displayed. Defaults to False.
        ad (int): Offset time to the base time for the ego vehicle to start in the simulation (only relevant in evaluation mode).
    """
    def __init__(self, config: configparser.ConfigParser, logger: logging.Logger, sumo_path: str, evaluate: bool=False, evaluator=None, gui: bool=False, ad: int=0) -> None:

        super().__init__()

        self.config = config
        self.logger = logger
        self.sumo_path = sumo_path
        self.evaluate = evaluate
        self.evaluator = evaluator
        self.gui = gui or self.config.getboolean('Simulation', 'gui')
        self.ad = ad if evaluate else random.randint(1, 7200)

        self.base = self.config.getint('Simulation', 'base_starttime')
        self.sumo_cfg = self.config.get('Simulation', 'sumo_cfg')
        self.vehicle_id = 'ego'
        self.emission_class = self.config.get('Simulation', 'vehicle_type')
        self.seed = self.config.get('Simulation', 'seed')

        # Choose SUMO interface: libsumo or traci
        if self.config.getboolean('Simulation', 'libsumo'):
            import libsumo as traci
        else:
            import traci
        self.traci = traci

        # Define the action space (agent outputs values between -1 and 1 )
        self.action_definition = ActionDefinition(config, self.logger, self.ad)
        # Define the simulation setup
        self._configure_simulation()

    def step(self, action: int, pre_eval: bool = False) -> bool:
        """
        Executes one step in the SUMO simulation using the given action. The simulation is stepped
        forward for a configured number of steps or until the ego vehicle is no longer present.
        If evaluation mode is active, performance metrics are gathered using the evaluator object.

        Args:
            action (int): Discrete action to be applied to the ego vehicle.
            pre_eval (bool): If True, performs a pre-evaluation action setup (baseline without ROSA).

        Returns:
            bool: True if the simulation is done (e.g., ego vehicle is no longer in the simulation), False otherwise.
        """

        while True:
            done = False

            # Check if the ego vehicle is currently in the simulation
            if self.vehicle_id in self.traci.vehicle.getIDList():
                # Apply the action to the ego vehicle
                self.action_definition.set_action(action, self.vehicle_id, pre_eval)
                # Advance the simulation for a fixed number of steps
                for step in range(self.steps):
                    # Stop if the ego vehicle has left the simulation
                    if not self.vehicle_id in self.traci.vehicle.getIDList():
                        done = True
                        break

                    # Update the GUI to track the ego vehicle (optional)
                    if self.gui:
                        pass
                        self.traci.gui.trackVehicle('View #0', self.vehicle_id)
                        self.traci.gui.setZoom('View #0', 500)

                    # Advance the simulation by one step
                    self.traci.simulation.step()

                    # If in evaluation mode, collect evaluation data
                    if self.evaluate & (self.vehicle_id in self.traci.vehicle.getIDList()):
                        self.evaluator.get_info()
                    # Stop if the ego vehicle disappeared during this step
                    if not self.vehicle_id in self.traci.vehicle.getIDList():
                        done = True
                        break
                break # Exit the outer while-loop once the step loop has run
            else:
                # Advance the simulation until the ego vehicle appears
                self.traci.simulation.step()

        # Close the SUMO connection if simulation is done
        if done:
            self.traci.close()

        return done

    def reset(self) -> None:
        """
        Resets the SUMO environment for a new episode.
        """

        self.logger.info('Resetting environment')
        # Generate ego vehicle and associated scenario
        self.generate_ego()
        # Prepare the SUMO startup command with current configuration
        self._sumo_cmd = configure_sumo(self.gui, self.sumo_path, self.sumo_cfg, self.ad + self.base - 180, self.seed)

        # Try starting the SUMO simulation, retrying if a connection issue occurs
        while True:
            try:
                self.traci.start(self._sumo_cmd, 8813) # Start SUMO with given command and port
                break  # Exit loop if TraCI connects successfully
            except self.traci.exceptions.TraCIException as e:
                # Log the error and attempt to close any potentially open connections
                self.logger.error(f'TraCIException encountered: {e}. Retrying...')
                self.traci.close()  # Ensure SUMO is closed before retrying

    def generate_ego(self) -> None:
        """
        Generates the ego vehicle with the specified route and departure time
        and inserts it into the SUMO route file.
        """

        self.logger.info(f'******************************** generating ego at {self.base + self.ad}')

        # If not evaluating, pick a random departure time offset between 1 and 7200 seconds
        if not self.evaluate:
            self.ad = random.randint(1, 7200)

        # Configure traffic setting
        route_file = self.config.get('Simulation', 'route_file')
        # Parse the route XML file
        tree = ET.parse(os.path.join(self.sumo_path, route_file))
        root = tree.getroot()

        # Validate emission class is either 'bev' or 'ice'
        assert self.emission_class in ['bev', 'ice'], 'Emission class must be either bev or ice'
        # Remove the default 'standard_car' vehicle type to replace with the specified emission class
        for vtype in root.findall('vType'):
            if vtype.attrib['id'] == 'standard_car':
                root.remove(vtype)
        # Insert the new vehicle type XML snippet based on emission class
        if self.emission_class == 'bev':
            new_vtype = ET.fromstring(bev)
        else:
            new_vtype = ET.fromstring(ice)
        root.insert(0, new_vtype)

        # Create the ego vehicle element
        vehicle = ET.Element("vehicle", id=self.vehicle_id, type='standard_car', departSpeed="max", departLane="1", sigma="0", depart=str(self.ad + self.base), color="0,1,0", IcKeepRight="0")
        # Create and append the route element to the vehicle
        route = ET.Element("route", edges=self.config.get('Simulation', 'ego_route'))
        vehicle.append(route)
        # Insert the ego vehicle into the route file in order of departure time
        for i, v in enumerate(root):
            if 'depart' in v.keys():
                if float(v.attrib['depart']) > (self.base + self.ad):
                    break
        root.insert(i, vehicle)

        # Write the updated route XML to a new file
        tree.write(os.path.join(self.sumo_path, "add_ego.rou.xml"))
        self.logger.info('finished generating ego')

    def _configure_simulation(self) -> None:
        """
        Configures the simulation environment by performing essential setup steps:
        """

        self.generate_ego() # Generate the ego vehicle and inserts it into the simulation.
        self.steps = self._set_step_len() # Set the simulation step length from the configuration.
        self._setup_action_space() # Define the action space for the agent.
        self._setup_simulation_environment() # Set up and starts the SUMO simulation environment.

    def _set_step_len(self) -> int:
        """
        Retrieves the simulation step length from the configuration file.

        Returns:
            int: Number of steps to execute per action in the simulation.
        """

        step_len = self.config.getint('General', 'steps')
        return step_len

    def _setup_action_space(self) -> None:
        """
        Defines the action space for the agent.
        Here, the action space is a continuous Box space with one dimension, bounded between -1 and 1.
        """

        # Create a 1-dimensional continuous action space with bounds [-1, 1]
        self.action_space = Box(low=np.array([-1 for _ in range(1)]), high=np.array([1 for _ in range(1)]))

    def _setup_simulation_environment(self) -> None:
        """
        Sets up the SUMO simulation environment.
        """

        self.steps = self.config.getint('General', 'steps')
        self.agent = self.config.get('General', 'agent')
        # Compose the SUMO command with GUI flag, paths, scenario parameters, and random seed
        self._sumo_cmd = configure_sumo(self.gui, self.sumo_path, self.sumo_cfg, self.ad + self.base - 180, self.seed)
        # Start SUMO simulation with the generated command on port 8813
        self.traci.start(self._sumo_cmd, 8813)


if __name__ == "__main__":
    raise NotImplementedError("This script is not intended to be run directly.")