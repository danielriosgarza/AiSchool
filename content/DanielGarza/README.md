# DanielGarza Microbiome Modeling Package

A comprehensive package for kinetic modeling and reinforcement learning applied to microbial community dynamics.

## Features

### Kinetic Modeling (`kinetic_model`)
- **Metabolite Management**: Define and track metabolite concentrations
- **Microbiome Simulation**: Model bacterial populations and their interactions
- **Reactor Control**: Simulate bioreactor environments with customizable parameters
- **Visualization**: Generate interactive plots and network diagrams
- **JSON Model Import**: Load pre-configured models from JSON files

### Reinforcement Learning (`rl`)
- **General RL Environment**: Config-driven, model-agnostic RL framework
- **Action Schemas**: Flexible action definitions for microbiome control
- **Reward Engineering**: Customizable reward functions for different objectives
- **Observation Builders**: Dynamic observation space construction
- **Training CLI**: Command-line tools for agent training and evaluation

## Installation

```bash
# Clone the repository
git clone https://github.com/danielriosgarza/AiSchool.git

# Navigate to the package directory
cd AiSchool/content/DanielGarza

# Install the package
pip install .
```

## Quick Start

### Kinetic Modeling

```python
from kinetic_model import Reactor, Metabolite, Bacteria, Microbiome

# Create a metabolite
glucose = Metabolite("glucose", initial_concentration=10.0)

# Create a bacterial population
ecoli = Bacteria("E. coli", growth_rate=0.5)

# Create a microbiome
microbiome = Microbiome([ecoli])

# Create and run a reactor
reactor = Reactor(microbiome, [glucose])
reactor.simulate(time_steps=100)
```

### Reinforcement Learning

```python
from rl import GeneralMicrobiomeEnv

# Create an RL environment
env = GeneralMicrobiomeEnv(config_path="rl/examples/configs/acetate_control.yaml")

# Train an agent (example with stable-baselines3)
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

## Model Templates

Pre-configured model templates are available in the `modelTemplates/` directory:
- `activated_sludge.json` - Activated sludge wastewater treatment
- `anaerobic_digestor.json` - Anaerobic digestion process
- `ecoli_overflow_simple.json` - Simple E. coli overflow metabolism
- `kombucha.json` - Kombucha fermentation process

## Documentation

Detailed documentation is available in the `rl/docs/` directory:
- `DOCUMENTATION.md` - Complete API reference
- `general_RL_short_tutorial.md` - RL framework tutorial
- `README.md` - RL package overview

## Requirements

- Python >= 3.9
- NumPy >= 1.21
- SciPy >= 1.8
- Plotly >= 5.0
- Gym >= 0.21
- Stable-Baselines3 >= 2.0
- PyYAML >= 6.0
- Matplotlib >= 3.5
- Pandas >= 1.4

## License

This project is licensed under the MIT License.

## Author

Daniel Rios Garza
