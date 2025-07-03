import configparser
import logging
import os
import pandas as pd
from typing import List

config = configparser.ConfigParser()
config.read("config.ini")

if config.getboolean('Simulation', 'libsumo'):
    import libsumo as traci
else:
    import traci

class ROSA_agent:
    """
    Reads prediction-based roundabout occupancy states and calculates advisory speeds for vehicles (ego) within the SUMO simulation.

    Args:
        config (configparser.ConfigParser): Configuration parser object with simulation settings.
        logger (logging.Logger): Logger instance for tracking experiment progress and issues.
        ad (int): Offset time to the base time for the ego vehicle to start in the simulation (only relevant in evaluation mode).
    """

    def __init__(self, config: configparser.ConfigParser, logger: logging.Logger, ad: int) -> None:

        self.config = config
        self.logger = logger
        self.ad = ad

        self.base = config.getint('Simulation', 'base_starttime')
        self.vehicle_id = 'ego'
        self.max_speed = config.getfloat('General', 'max_speed')
        self.min_speed = config.getfloat('General', 'min_speed')
        self.max_deceleration = self.config.getfloat('General', 'max_deceleration')

        # Load the Excel file containing prediction data
        occupancy_states = pd.ExcelFile(os.path.join(config.get('Prediction', 'model_path'), config.get('Prediction', 'occupancy_file')))
        # Select the appropriate prediction columns based on configuration
        if config.get('Prediction', 'prediction') == 'target':
            self.prediction_vru = 'state_target_vru'
            self.prediction_veh = 'state_target_veh'

        elif config.get('Prediction', 'prediction') == 'output':
            self.prediction_vru = 'state_output_vru'
            self.prediction_veh = 'state_output_veh'

        # Load VRU and vehicle prediction data from the sheet "Prediction_5"
        self.vru_phases = pd.read_excel(occupancy_states, sheet_name="Prediction_5", usecols=[self.prediction_vru])
        self.veh_phases = pd.read_excel(occupancy_states, sheet_name="Prediction_5", usecols=[self.prediction_veh])
        self.vru_phases.index = self.vru_phases.index.astype(int)
        self.veh_phases.index = self.veh_phases.index.astype(int)

        # Internal state to track whether advisory speeds have already been initialized
        self.advisory_speed_vru_initialized = False
        self.advisory_speed_veh_initialized = False

        self.advisory_speed_vru = None
        self.advisory_speed_veh = None
        self.advisory_speed = self.max_speed
        self.deceleration = self.config.getfloat('General', 'max_deceleration')

        # Choose SUMO interface: libsumo or traci
        if self.config.getboolean('Simulation', 'libsumo'):
            import libsumo as traci
        else:
            import traci
        self.traci = traci

    def predict(self) -> List[float]:

        """
        Computes the advisory speed and deceleration for the ego vehicle
        based on predicted occupancy states for both VRUs and vehicles, and distances to stop lines.

        Returns:
            action (List[float]): A list containing:
                - advisory_speed (float): The recommended speed for the ego vehicle.
                - deceleration (float): The corresponding deceleration to be applied.
        """

        if self.vehicle_id in traci.vehicle.getIDList():

            # Get the distance already travelled by the ego vehicle
            distance = traci.vehicle.getDistance(self.vehicle_id)
            self.logger.info(f'Travelled distance ego: {round(distance, 2)}')

            # Attempt to retrieve the distances to the traffic lights representing VRU and vehicle occupancy
            try:
                tl_distance_veh = traci.vehicle.getNextTLS(self.vehicle_id)[1][2]
                tl_distance_vru = traci.vehicle.getNextTLS(self.vehicle_id)[0][2]
                self.logger.info(f'distance_sl vru: {round(tl_distance_vru, 2)}')
                self.logger.info(f'distance_sl veh: {round(tl_distance_veh, 2)}')
            except:
                tl_distance_veh = -100
                tl_distance_vru = -100

            # Calculate the time difference since the ego vehicle entered the simulation
            current_time = int(traci.simulation.getTime())
            time_diff = current_time - (self.base + self.ad)
            #self.logger.info(f'Time difference: {time_diff}')

            current_speed = traci.vehicle.getSpeed(self.vehicle_id)
            self.logger.info(f'Speed: {round(current_speed, 2)}')

            # Time to stop line estimations based on initial predefined values
            tsl_vru_init = 22
            tsl_veh_init = 24
            tsl_vru = tsl_vru_init - time_diff
            tsl_veh = tsl_veh_init - time_diff

            if tsl_veh > 0:
                self.logger.info(f'Time-to-stopline vru: {tsl_vru}')
                self.logger.info(f'Time-to-stopline veh: {tsl_veh}')

            # Apply ROSA logic if certain conditions are met
            if (current_speed >= 1) & (distance < 269) & (tl_distance_veh > 0) & (tsl_vru <= 5):
                self.logger.info(f'self.advisory_speed_vru_initialized: {self.advisory_speed_vru_initialized}')

                # --- Advisory Speed Calculation for VRU Occupancy ---
                if not self.advisory_speed_vru_initialized:
                    index_vru = int(self.ad + tsl_vru_init -4)
                    self.logger.info(f'index vru: {index_vru}')
                    reaching_state_5s_vru = self.vru_phases.loc[index_vru, self.prediction_vru]
                    self.logger.info(f'reaching_state_5s_vru: {reaching_state_5s_vru}')

                    self.advisory_speed = self.calculate_speedAdvisory(reaching_state_5s_vru, tsl_vru, tl_distance_vru, self.max_speed, self.min_speed)
                    self.advisory_speed_vru_initialized = True
                    self.logger.info(f'1 - Advisory speed: {self.advisory_speed}')

                else:
                    if self.advisory_speed != self.max_speed:
                        tsl_vru_new = round(tl_distance_vru / current_speed)
                        self.logger.info(f'New time-to-stopline vru: {tsl_vru_new}')
                        if (tsl_vru_new <= 5):
                            index_vru = int(current_time + tsl_vru_new - self.base - 4)
                            self.logger.info(f'index vru: {index_vru}')
                            reaching_state_5s_vru = self.vru_phases.loc[index_vru, self.prediction_vru]
                            self.logger.info(f'reaching_state_5s_vru: {reaching_state_5s_vru}')

                            if reaching_state_5s_vru == 'g':
                                pass
                            else:
                                self.advisory_speed = self.calculate_speedAdvisory(reaching_state_5s_vru, tsl_vru_new,
                                                                          tl_distance_vru,
                                                                          self.max_speed, self.min_speed)
                            self.logger.info(f'2 - Advisory speed: {self.advisory_speed}')
                    else:
                        pass

                # --- Advisory Speed Calculation for Vehicle Occupancy ---
                if self.advisory_speed != self.max_speed:
                    tsl_veh_new = round(tl_distance_veh / current_speed)
                else:
                    tsl_veh_new = tsl_veh
                self.logger.info(f'New time-to-stopline veh: {tsl_veh_new}')
                if (tsl_veh_new <= 5):
                    if not self.advisory_speed_veh_initialized:
                        self.logger.info(f'self.advisory_speed_veh_initialized: {self.advisory_speed_veh_initialized}')
                        index_veh = int(current_time+tsl_veh_new -self.base - 4)
                        self.logger.info(f'index veh: {index_veh}')
                        reaching_state_5s_veh = self.veh_phases.loc[index_veh, self.prediction_veh]
                        self.logger.info(f'reaching_state_5s_veh: {reaching_state_5s_veh}')

                        if reaching_state_5s_veh == 'g':
                            pass
                        else:
                            self.advisory_speed = self.calculate_speedAdvisory(reaching_state_5s_veh, tsl_veh_new, tl_distance_veh, self.max_speed, self.min_speed)
                        self.advisory_speed_veh_initialized = True
                        self.logger.info(f'3 - Advisory speed: {self.advisory_speed}')

                    else:
                        if self.advisory_speed != self.max_speed:
                            tsl_veh_new = round(tl_distance_veh / current_speed)
                            self.logger.info(f'New time-to-stopline veh: {tsl_veh_new}')

                            if (tsl_veh_new <= 5):
                                index_veh = int(current_time + tsl_veh_new - self.base - 4)
                                self.logger.info(f'index veh: {index_veh}')
                                reaching_state_5s_veh = self.veh_phases.loc[index_veh, self.prediction_veh]
                                self.logger.info(f'reaching_state_5s_veh: {reaching_state_5s_veh}')

                                if reaching_state_5s_veh == 'g':
                                    pass
                                else:
                                    self.advisory_speed = self.calculate_speedAdvisory(reaching_state_5s_veh,
                                                                                  tsl_veh_new,
                                                                                  tl_distance_veh,
                                                                                  self.max_speed, self.min_speed)
                                self.logger.info(f'4 - Advisory speed: {self.advisory_speed}')
                        else:
                            pass

            else:
                # If no prediction logic is triggered, use default speed and deceleration
                self.advisory_speed = self.max_speed

        if self.advisory_speed == self.max_speed:
            self.deceleration = self.max_deceleration
        else:
            self.deceleration = 2.0

        self.logger.info(f'Advisory: {self.advisory_speed}')

        return [self.advisory_speed, self.deceleration]

    def calculate_speedAdvisory(self, reaching_state: str, tsl: int, distance_diff: float, max_speed: float, min_speed: float ) -> float:
        """
        Calculates the optimal advisory speed for a vehicle approaching a vehicle/VRU conflict zone.

        This function computes a recommended speed based on the current occupancy state
        the vehicle is expected to reach ("g" for free, "r" for occupied), the time until the vehicle
        would arrive at the stopline (TSL), and the remaining distance. If the zone will be occupied
        upon arrival, the function estimates whether it is feasible to decelerate and reach the next
        free state at a reasonable speed.

        Args:
            reaching_state (str): Expected occupancy state when reaching the stopline ('g' or 'r').
            tsl (int): Time (in seconds) until the vehicle reaches the stopline.
            distance_diff (float): Distance (in meters) from the vehicle to the stopline.
            max_speed (float): Maximum allowed advisory speed.
            min_speed (float): Minimum allowed advisory speed.

        Returns:
            float: Calculated advisory speed in m/s.
        """

        # Estimate time at which the ego vehicle should reach the stopline after applying ROSA (occupancy duration not known, therefore 1s later than without ROSA)
        new_arrival = tsl+1

        # Determine advisory speed based on predicted occupancy state
        if reaching_state == "g":
            # If the vehicle reaches a free state, it can proceed at max speed
            advisory_speed = max_speed

        elif reaching_state == "r":
            # If the vehicle reaches an occupied zone, check if it can decelerate to hit the next free state
            if new_arrival > 0:
                ttg_speed = round(distance_diff / new_arrival, 3) # Time-to-green speed
                #self.logger.info(f'Time-to-green-speed: {ttg_speed}')

                # If this speed is within allowed bounds, use it; otherwise use min_speed
                if ttg_speed <= max_speed:
                    advisory_speed = max(ttg_speed, min_speed)
                else:
                    advisory_speed = min_speed
            else:
                # If new_arrival is invalid or negative, advise max speed to avoid delay
                advisory_speed = max_speed

        return advisory_speed


if __name__ == "__main__":
    raise NotImplementedError("This script is not intended to be run directly.")