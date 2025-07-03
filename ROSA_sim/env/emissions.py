import configparser
config = configparser.ConfigParser()
config.read('config.ini')

ice_emissionclass = config.get('Simulation', 'ice_emissionclass')

"""
This file defines the standard car for the simulation. This will be used to adapt the emission class of the vehicle.
"""


bev = """<vType id="standard_car" accel="0.8" decel="0.8" length="5.0" maxSpeed='13.89' lcSpeedGainRight="0" sigma="0" emissionClass="Energy/unknown">
    <param key="has.battery.device" value="true"/>
    <param key="airDragCoefficient" value="0.35"/>
    <param key="constantPowerIntake" value="100"/>
    <param key="frontSurfaceArea" value="2.6"/>
    <param key="rotatingMass" value="40"/>
    <param key="maximumBatteryCapacity" value="64000"/>
    <param key="maximumPower" value="150000"/>
    <param key="propulsionEfficiency" value=".98"/>
    <param key="radialDragCoefficient" value="0.1"/>
    <param key="recuperationEfficiency" value=".96"/>
    <param key="rollDragCoefficient" value="0.01"/>
    <param key="stoppingThreshold" value="0.1"/>
    <param key="vehicleMass" value="1830"/>
</vType>"""

ice = f"""<vType id="standard_car" accel="0.8" decel="0.8" length="5.0" maxSpeed='13.89' lcSpeedGainRight="0" sigma="0" emissionClass="{ice_emissionclass}">
    <param key="airDragCoefficient" value="0.35"/>
    <param key="frontSurfaceArea" value="2.6"/>
    <param key="rotatingMass" value="40"/>
    <param key="radialDragCoefficient" value="0.1"/>
    <param key="rollDragCoefficient" value="0.01"/>
    <param key="stoppingThreshold" value="0.1"/>
    <param key="vehicleMass" value="1830"/>
</vType>"""



