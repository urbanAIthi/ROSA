import configparser

def validate_config(config: configparser.ConfigParser) -> None:
    """
    Validates the config file by checking types and allowed values.
    
    Args:
        config (configparser.ConfigParser): Configuration parser object with simulation settings.
    """
    
    def validate_bool(section: str, option: str) -> None:
        if config.get(section, option) not in ['True', 'False']:
            raise ValueError(f"Invalid boolean value for {section}/{option}: {config.get(section, option)}")

    def validate_int(section: str, option: str, min_value=None, max_value=None) -> None:
        try:
            value = int(config.get(section, option))
            if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                raise ValueError(f"Value for {section}/{option} must be between {min_value} and {max_value}")
        except ValueError:
            raise ValueError(f"Invalid integer value for {section}/{option}: {config.get(section, option)}")

    def validate_float(section: str, option: str, min_value=None, max_value=None) -> None:
        try:
            value = float(config.get(section, option))
            if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                raise ValueError(f"Value for {section}/{option} must be between {min_value} and {max_value}")
        except ValueError:
            raise ValueError(f"Invalid float value for {section}/{option}: {config.get(section, option)}")

    def validate_str(section: str, option: str, allowed_values=None) -> None:
        value = config.get(section, option)
        if allowed_values and value not in allowed_values:
            raise ValueError(f"Invalid value for {section}/{option}: {value}. Must be one of {allowed_values}")


    def validate_list_of_int(section: str, option: str) -> None:
        raw_value = config.get(section, option)
        value = raw_value.strip()[1:-1].split(',')
        value = [v.strip() for v in value if v.strip() != '']

        try:
            [int(v) for v in value]
        except ValueError:
            raise ValueError(f"Invalid list of integers for {section}/{option}: {raw_value}")



    # Validate [Simulation] section
    validate_bool('Simulation', 'gui')
    validate_bool('Simulation', 'libsumo')
    validate_int('Simulation', 'seed')
    validate_int('Simulation', 'base_starttime', min_value=0)
    validate_str('Simulation', 'sumo_cfg')
    validate_str('Simulation', 'vehicle_type', allowed_values=['ice', 'bev'])

    if config.get('Simulation', 'vehicle_type') == 'ice':
        validate_str('Simulation', 'ice_emissionclass', allowed_values=[
            'HBEFA4/PC_petrol_Euro-5', 'HBEFA4/PC_diesel_Euro-4',
            'HBEFA4/PC_petrol_Euro-4', 'HBEFA4/PC_diesel_Euro-5',
            'HBEFA3/PC_D_EU4', 'HBEFA3/PC_G_EU4', 'HBEFA3/PC_D_EU5',
            'HBEFA3/PC_G_EU5'
        ])

    validate_str('Simulation', 'ego_route')  # Assuming any string is valid for ego_route
    validate_str('Simulation', 'route_file')  # Assuming any string is valid for route_file
    validate_list_of_int('Simulation', 'evaluations')

    # Validate [General] section
    validate_str('General', 'agent', allowed_values=['classic'])
    validate_int('General', 'steps', min_value=1)
    validate_float('General', 'min_speed', min_value=0.0)
    validate_float('General', 'max_speed', min_value=0.0)
    validate_float('General', 'max_deceleration', min_value=0.0)
    validate_float('General', 'max_acceleration', min_value=0.0)

    # Validate [Prediction] section
    validate_str('Prediction', 'model_path') # Assuming any string is valid for model_path
    validate_str('Prediction', 'occupancy_file') # Assuming any string is valid for occupancy_file
    validate_str('Prediction', 'prediction') # Assuming any string is valid for prediction

    # Validate [wandb] section
    validate_str('wandb', 'project')
    validate_str('wandb', 'mode', allowed_values=['online', 'disabled'])


if __name__ == "__main__":
    raise NotImplementedError("This script is not intended to be run directly.")