# ROSA - The Roundabout Optimized Speed Advisory

This repository contains the official implementation for the paper:
**“ROSA: Roundabout Optimized Speed Advisory with Multi-Agent Trajectory Prediction in Multimodal Traffic”**,  
to be presented at **IEEE ITSC 2025**.

ROSA combines **multi-agent trajectory prediction** with **coordinated speed guidance** for vehicles approaching and entering roundabouts in **multimodal and mixed traffic environments**. It supports both automated and human-driven vehicles and provides proactive, real-time speed advisories based on predicted conflicts with other vehicles and Vulnerable Road Users (VRUs).


## Installation

### Prerequisites

- [Anaconda or Miniconda](https://www.anaconda.com/distribution/)

### Cloning the repository

```bash
git clone git@github.com:urbanAIthi/ROSA.git
cd ROSA
```

### Setting up the Conda environment

Use the following command to create a Conda environment based on the `environment.yaml` file:

```bash
conda env create -f environment.yaml --name rosa
conda activate rosa
```

### Installing SUMO

This repository requires that you already have installed the SUMO traffic simulator. For more information on how to install SUMO, please refer [to https://sumo.dlr.de/docs/Installing/index.html].
This project uses SUMO version 1.22.0. Please be aware that simulation results may vary when using different SUMO versions due to changes or updates in the simulator.


## Project Structure

This repository is organized into two main components, reflecting the architecture of the ROSA system as described in the paper.

### 1. `ROSA_pred/` – Transformer-Based Multi-Agent Trajectory Prediction

This module implements a Transformer-based model that jointly predicts the trajectories of all agents (vehicles and VRUs) at roundabouts.

- **Model architecture**: Transformer Encoder with attention masking to capture agent dynamics and inter-agent dependencies.
- **Input features**: Position, velocity, acceleration (lateral/tangential), heading angle (sin/cos), agent type, and optional exit intention.
- **Training**: Supervised single-step prediction based on 3-second history.
- **Inference**: Autoregressive prediction over a 5-second horizon.
- **Output**: Predicted trajectories including kinematic state (position, speed, acceleration, orientation).
- **Evaluation**: ADE/FDE and binary classification for occupancy of crosswalks and entries.
  
#### How to run

```bash
cd ROSA_pred
python main.py
```

The prediction settings (model parameters, training options, etc.) can be modified in train_configs.yaml.

#### Structure

ROSA_pred/  
  ├── configs/ # dir containing config files  
  ├── rdb1/ # dir containing roundabout specific files like dataset and shapefiles from openDD  
  ├── trained_models/ # dir containing models trained for the ROSA paper
  ├── utils/ # dir containing different utility functions  
  ├── generate_dataset.py # file for dataset preprocessing  
  ├── models.py # file for model handling  
  ├── main.py # main file for starting the preprocessing-training-testing pipeline
  ├── trainer.py # file for training the prediction model
  ├── tester.py # file for testing the prediction model
  

### 2. `ROSA_sim/` – Speed Advisory and SUMO Simulation

This module implements the ROSA speed advisory algorithm and evaluates it using realistic roundabout scenarios modeled in SUMO (Simulation of Urban MObility).

- **Goal**: Improve safety and efficiency by proactively adjusting vehicle speed based on predicted crosswalk and entry occupancy (derived from the ROSA trajectory prediction module).
- **Functionality**: Model-based calculation of optimal speed.
- **Evaluation**: 6,600 scenarios from the openDD dataset modeled in SUMO, considering the following metrics: fuel/energy consumption, emissions, travel time, waiting time, and stops.

#### How to run

```bash
cd ROSA_sim
python run.py
```

The simulation settings (scenarios, vehicle type, etc.) can be modified in config.ini.

#### Structure

ROSA_sim/  
  ├── env/ # dir containing simulation environment files (evaluation, speed advisory)
  ├── sumo/ # dir containing sumo configuration files
  ├── utils/ # dir containing different utility functions  
  ├── config.ini # file for simulation settings
  ├── run.py # main file for starting the simulation
  

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{ROSA_Schlamp,
  author={Schlamp, Anna-Lena and Gerner, Jeremias and Bogenberger, Klaus and Huber, Werner and Schmidtner, Stefanie},
  booktitle={2025 IEEE International Conference on Intelligent Transportation Systems (ITSC)}, 
  title={ROSA: Roundabout Optimized Speed Advisory with Multi-Agent Trajectory Prediction in Multimodal Traffic}, 
  year={2025},
  volume={},
  number={},
  pages={},
  doi={}
}
```

The Transformer implementation is based on the following repository: [https://github.com/urbanAIthi/FloatingCarObserver]
and the associated papers:
```bibtex
@INPROCEEDINGS{10422398,
   author={Gerner, Jeremias and Rößle, Dominik and Cremers, Daniel and Bogenberger, Klaus and Schön, Torsten and Schmidtner, Stefanie},
   booktitle={2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC)}, 
   title={Enhancing Realistic Floating Car Observers in Microscopic Traffic Simulation}, 
   year={2023},
   pages={2396-2403},
   doi={10.1109/ITSC57777.2023.10422398}
}
```
```bibtex
@article{gerner2025FCO_TFCO,
  author       = {Jeremias Gerner and Klaus Bogenberger and Stefanie Schmidtner},
  title        = {Floating Car Observers in Intelligent Transportation Systems: Detection Modeling and Temporal Insights},
  journal      = {arXiv},
  year         = {2025},
}
```

The openDD trajectory dataset is used for this work: [https://l3pilot.eu/data/opendd.html].


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
