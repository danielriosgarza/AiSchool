# `mg_rl_general`: A General Reinforcement Learning Framework for Microbiome Control

## Table of Contents
- [Overview](#overview)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
  - [Episode (`episode`)](#episode-episode)
  - [Simulation (`simulation`)](#simulation-simulation)
  - [Actions (`actions`)](#actions-actions)
  - [Observations (`observations`)](#observations-observations)
  - [Rewards (`rewards`)](#rewards-rewards)
  - [Initial Randomization (`init_randomization`)](#initial-randomization-init_randomization)
- [Environment (`GeneralMicrobiomeEnv`)](#environment-generalmicrobiomeenv)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
  - [Methods](#methods)
- [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [`train_sac.py`](#train_sacpy)
  -[`evaluate.py`](#evaluatepy)
  - [`random_baseline.py`](#random_baselinepy)
  - [`plot_general_controller.py`](#plot_general_controllerpy)
- [Key Concepts](#key-concepts)
  - [Observation Pipeline](#observation-pipeline)
  - [Expression-based Rewards and Observations](#expression-based-rewards-and-observations)
  - [Model Adapter](#model-adapter)
- [Complete Example](#complete-example)
- [Testing](#testing)

## Overview

The `mg_rl_general` package provides a configurable, model-agnostic Reinforcement Learning (RL) environment for controlling microbiome simulations. It is built on top of the `mg_kinetic_model` package and uses `stable-baselines3` for the underlying RL algorithms.

The framework is designed to be flexible and extensible, allowing users to define custom observation and reward schemes using simple expressions. It supports both emergent and controlled pH, and provides command-line interfaces for training, evaluation, and running baselines.

### Key Features:
- **Config-driven**: The entire RL environment, including the agent's actions, observations, and rewards, is defined by a single YAML or JSON configuration file.
- **Model-agnostic**: The framework can be used with any microbiome model that can be exported to the format expected by `mg_kinetic_model`.
- **Expression-based Rewards and Observations**: Define complex reward functions and observation features using a simple expression language, without needing to write any code.
- **Observation Pipeline**: A flexible pipeline for processing observations, including normalization, clipping, and stacking historical frames.
- **Command-Line Tools**: Simple CLI scripts for training new models, evaluating existing ones, and comparing against a random baseline.
- **Support for different pH modes**: The environment can be configured to simulate a system with a controlled pH, an emergent pH, or allow the agent to switch between the two.

## Quickstart

This section provides a step-by-step guide to get you started with training and evaluating a model.

### Training a Model
You can train a Soft Actor-Critic (SAC) model using the `train_sac.py` script. You need to provide a path to the kinetic model JSON file and a configuration file.

An example configuration is provided in the `examples/configs` directory.

```bash
python backend/src/mg_rl_general/cli/train_sac.py \
    --model-json backend/examples/bh_bt_ri_complete_model_export.json \
    --config backend/src/mg_rl_general/examples/configs/emergent_ph_minimal.yaml \
    --timesteps 10000 \
    --n-envs 4 \
    --eval-freq 2000 \
    --log-dir logs/mg_rl_general \
    --model-path models/mg_rl_general_sac
```

This command will train a SAC model for 10,000 timesteps using 4 parallel environments. The model will be saved to `models/mg_rl_general_sac.zip`.

### Evaluating a Trained Model
Once you have a trained model, you can evaluate its performance using the `evaluate.py` script.

```bash
python backend/src/mg_rl_general/cli/evaluate.py \
    --model-json backend/examples/bh_bt_ri_complete_model_export.json \
    --config backend/src/mg_rl_general/examples/configs/emergent_ph_minimal.yaml \
    --model-path models/mg_rl_general_sac.zip
```
This will run the trained agent in the environment and print the evaluation results.

### Running a Random Baseline
To compare your agent's performance against a random agent, you can use the `random_baseline.py` script.

```bash
python backend/src/mg_rl_general/cli/random_baseline.py \
    --model-json backend/examples/bh_bt_ri_complete_model_export.json \
    --config backend/src/mg_rl_general/examples/configs/emergent_ph_minimal.yaml \
    --episodes 3
```
This will run a random agent for 3 episodes and print the results.

### Horizon and Episode Mix (Restricted Parity)

The restricted trainer supports short/long episode mixing (e.g., 10 vs 250 steps per episode). The general environment now supports mixing via `episode.mix`.

Examples:

```yaml
episode:
  mix:
    enabled: true
    short_horizon: 10
    long_horizon: 250
    short_prob: 0.3
  dt_hours: 1.0
  training_mode: balanced
```

Notes:
- When `mix.enabled` is true, the environment samples the episode type at `reset()` and uses the corresponding horizon.
- You can force the next episode type via `env.reset(options={"episode_type": "short"|"long"})`.
- Each episode truncates exactly at its sampled horizon.

## Configuration

The behavior of the `GeneralMicrobiomeEnv` and the training process is controlled by a single configuration file. This file is parsed into the `TopLevelConfig` Pydantic model. Below is a detailed explanation of each section of the configuration.

### Episode (`episode`)

This section configures the properties of an episode.

- **`horizon`** (`int`, default: `250`): The maximum number of steps in an episode.
- **`dt_hours`** (`float`, default: `1.0`): The duration of a single timestep in hours.
- **`training_mode`** (`str`, default: `"balanced"`): The simulation mode to use during training. Can be one of `"fast"`, `"balanced"`, or `"accurate"`. This controls the trade-off between simulation speed and accuracy.
- **`randomize_horizon`** (`bool`, default: `False`): If `True`, the horizon for each episode will be randomly sampled from the `horizon_range`.
- **`horizon_range`** (`Tuple[int, int]`, default: `(100, 250)`): The range of possible horizon values when `randomize_horizon` is `True`.

### Simulation (`simulation`)

This section configures the underlying kinetic model simulation.

- **`min_steps_per_pulse`** (`int`, default: `10`): The minimum number of integration steps to take per RL step (pulse).
- **`steps_per_hour_factor`** (`int`, default: `50`): A factor that scales the number of integration steps per hour. The total number of integration steps per RL step is `dt_hours * steps_per_hour_factor`.

### Actions (`actions`)

This section defines the action space of the RL agent.

- **`pH_mode`** (`str`, default: `"switchable"`): Determines how the pH is handled in the environment.
  - `"controlled"`: The agent directly controls the pH by setting a target value (`pH_set`).
  - `"emergent"`: The pH emerges from the concentrations of metabolites in the reactor. The agent does not directly control it.
  - `"switchable"`: The agent can choose to either control the pH or let it be emergent at each timestep, using the `pH_ctrl` action.

- **`actuators`** (`List[Dict]`, default: see `config.py`): A list of dictionaries, where each dictionary defines an actuator available to the agent. Each actuator has a `name` and a `type`.
  - **`name`** (`str`): The name of the actuator. The standard actuators are `q`, `v`, `pH_ctrl`, `pH_set`, `stir`, and `temp`.
  - **`type`** (`str`): The type of the actuator. Can be `"continuous"` or `"binary"`.

- **`bounds`** (`Dict`, default: see `ActionBounds` in `config.py`): A dictionary that specifies the lower and upper bounds for each of the continuous actuators.
  - `q`: Bounds for the feed rate.
  - `v`: Bounds for the valve/output rate.
  - `pH_set`: Bounds for the pH setpoint.
  - `stir`: Bounds for the stirring rate.
  - `temp`: Bounds for the temperature.
  - `pH_ctrl_threshold`: The threshold for the binary `pH_ctrl` action when using a continuous action space.

- **`smoothness`** (`Dict`, default: `{"enabled": False, "max_delta": {}}`): Configuration for action smoothness constraints.
  - **`enabled`** (`bool`): If `True`, enforces a maximum change in action values between timesteps.
  - **`max_delta`** (`Dict[str, float]`): A dictionary mapping actuator names to their maximum allowed change per timestep.

### Observations (`observations`)

This section defines the observation space of the RL agent.

- **`include`** (`List[str]`, default: `["met.all", "pH.used", "actuator_echo.all", "met.delta.all", "met.rate.all"]`): A list of strings that specify which features to include in the observation space. The available features are:
  - `"met.all"`: The concentrations of all metabolites.
  - `"pH.used"`: The pH of the reactor.
  - `"actuator_echo.all"`: The previous action taken by the agent.
  - `"met.delta.all"`: The change in metabolite concentrations from the previous timestep.
  - `"met.rate.all"`: The rate of change of metabolite concentrations.
  - `"expr:<expression>"`: A custom observation defined by an expression. See [Expression-based Rewards and Observations](#expression-based-rewards-and-observations).

- **`population`** (`Dict`): Configuration for including population-level features in the observation. This is broken down into `species` and `subpopulations`.
  - **`species`**: Configuration for species-level observations.
    - **`mode`**: Can be `"off"`, `"live"`, `"aggregates"`, or `"summary"`.
    - **`signals`**: Can be `"live_count"` or `"live_share"`.
    - **`normalization`**: Can be `"sum1"`, `"log1p"`, or `"clr"`.
  - **`subpopulations`**: Configuration for subpopulation-level observations.
    - **`mode`**: Can be `"off"`, `"dense"`, or `"tokens"`.

- **`pipeline`** (`List[Dict]`, default: `[{"normalize": {"method": "running_mean_var"}}, {"clip": {"min": -10, "max": 10}}]`): A list of processing steps to apply to the observation vector. See [Observation Pipeline](#observation-pipeline) for more details.

### Rewards (`rewards`)

This section defines the reward function for the RL agent.

- **`error_reward`** (`float`, default: `-1000.0`): The reward given to the agent when the simulation encounters an error (e.g., non-finite state).

- **`terms`** (`List[Dict]`): A list of dictionaries, where each dictionary defines a term in the step reward function. Each term has an `expr`, a `weight`, and an optional `deadband`.
  - **`expr`** (`str`): The expression to evaluate for this reward term. See [Expression-based Rewards and Observations](#expression-based-rewards-and-observations) for the available syntax.
  - **`weight`** (`float`): The weight to multiply the result of the expression by.
  - **`deadband`** (`float`, optional): A deadband around zero for the expression result. If the absolute value of the result is less than the deadband, it is treated as zero.

- **`terminal`** (`List[Dict]`): A list of dictionaries defining terminal rewards, which are given only on the last step of an episode.
  - **`when`** (`str`): When to apply the terminal reward. Currently, only `"last_step"` is supported.
  - **`expr`** (`str`): The expression to evaluate for the terminal reward.
  - **`weight`** (`float`): The weight for the terminal reward.

### Initial Randomization (`init_randomization`)

This section allows for the randomization of the initial state of the environment at the beginning of each episode. This can be useful for training a more robust agent.

- **`enabled`** (`bool`, default: `False`): Set to `True` to enable initial state randomization.

- **`apply_to`** (`str`, default: `both`): Limit randomization to specific episode types when mixing is enabled. One of `short`, `long`, or `both`.

- **`metabolites`** (`List[Dict]`): A list of rules for randomizing initial metabolite concentrations. Each rule has a `pattern`, `low`, and `high`.
  - **`pattern`** (`str`): A regular expression that is matched against metabolite names.
  - **`low`** (`float`): The lower bound for the random concentration.
  - **`high`** (`float`): The upper bound for the random concentration.

- **`subpopulations`** (`List[Dict]`): A list of rules for randomizing initial subpopulation states. Each rule has a `state`, `low`, and `high`.
  - **`state`** (`str`): The state of the subpopulation to randomize (e.g., `"active"`).
  - **`low`** (`float`): The lower bound for the random value.
  - **`high`** (`float`): The upper bound for the random value.

## Environment (`GeneralMicrobiomeEnv`)

The `GeneralMicrobiomeEnv` is a `gymnasium.Env` that wraps the microbiome simulation. It is instantiated with a path to a model JSON file and a `TopLevelConfig` object.

### Action Space

The action space is a `gymnasium.spaces.Box` with shape `(N,)`, where `N` is the number of continuous actuators. The values are in the range `[-1, 1]`. The environment internally scales these actions to the bounds specified in the configuration.

The number of dimensions in the action space depends on the `pH_mode` in the `actions` configuration:
- If `pH_mode` is `"emergent"`, the action space has 4 dimensions, corresponding to `q`, `v`, `stir`, and `temp`.
- If `pH_mode` is `"controlled"` or `"switchable"`, the action space has 6 dimensions, corresponding to `q`, `v`, `pH_ctrl`, `pH_set`, `stir`, and `temp`.

### Observation Space

The observation space is a `gymnasium.spaces.Box` with shape `(M,)`, where `M` is the total number of features in the observation, as defined by the `observations` section of the configuration. The space has bounds `[-inf, inf]`, but the `clip` wrapper in the observation pipeline is typically used to constrain the values.

The shape of the observation space is determined at the first call to `reset()`.

### Methods

- **`reset(*, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]`**: Resets the environment to an initial state.
  - **Returns**: A tuple containing the initial observation and an info dictionary. The info dictionary contains the observation manifest (`obs_manifest`), which describes the features in the observation vector.

- **`step(action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]`**: Steps the environment forward one timestep.
  - **Arguments**:
    - `action`: The action to take, as a NumPy array.
  - **Returns**: A tuple containing:
    - `observation`: The observation for the new state.
    - `reward`: The reward for the timestep.
    - `terminated`: A boolean indicating if the episode has terminated.
    - `truncated`: A boolean indicating if the episode has been truncated (e.g., reached the horizon).
    - `info`: An info dictionary containing debugging information, such as the reward breakdown (`reward_breakdown`) and KPIs (`kpis_partial`).

## Command-Line Interface (CLI)

The `mg_rl_general` package includes several command-line scripts for training, evaluating, and testing models.

### `train_sac.py`

This script trains a Soft Actor-Critic (SAC) model on the `GeneralMicrobiomeEnv`.

**Usage:**
```bash
python backend/src/mg_rl_general/cli/train_sac.py [OPTIONS]
```

**Arguments:**

- `--model-json`: (str) Path to the kinetic model JSON file.
- `--config`: (str) Path to the YAML/JSON configuration file for the environment.
- `--timesteps`: (int, default: 5000) The total number of timesteps to train for.
- `--n-envs`: (int, default: 1) The number of parallel environments to use for training.
- `--eval-freq`: (int, default: 2000) The frequency (in timesteps) at which to evaluate the model.
- `--log-dir`: (str) The directory to save the training logs.
- `--model-path`: (str) The path to save the trained model.
- `--seed`: (int, default: 42) The random seed for reproducibility.
- `--save-freq`: (int, default: 0) The frequency (in timesteps) at which to save model checkpoints. If 0, no checkpoints are saved.
- `--save-norm-with-checkpoints`: (flag) When used with `--save-freq > 0`, also saves observation normalization stats JSON next to each checkpoint (filename suffix `_ckpt_<steps>_normalization_stats.json`).
- `--freeze-norm-steps`: (int, default: 0) The number of timesteps after which to freeze the observation normalization. If 0, normalization is never frozen.
- `--learning-rate`: (float, default: 3e-4) The learning rate for the SAC optimizer.
- `--lr-schedule`: (str, default: "constant") The learning rate schedule. Can be "constant" or "linear".
- `--ent-coef`: (str, default: "auto_0.5") The entropy coefficient for the SAC algorithm.
- `--train-freq`: (int, default: 1) Number of environment rollout steps collected before a training phase. Together with `--gradient-steps`, it sets the update ratio: updates per env step = `gradient_steps / train_freq` (after `--learning-starts`).
- `--gradient-steps`: (int, default: 1) Number of gradient updates performed during each training phase triggered by `--train-freq`.
- `--progress-bar`: (flag) Show a tqdm progress bar during training (SB3 >= 2.0).
- `--tensorboard-log`: (str, optional) Directory to write TensorBoard logs (enables SB3 TensorBoard logging).
- `--tb-log-name`: (str, default: `sac`) Run name (subfolder) under `--tensorboard-log`.
- `--entropy-target-at`: (str) A comma-separated list of `step:target` pairs for scheduling the entropy target. Example: `60000:-4,120000:-6`.
- `--sde-freeze-at`: (int, default: 0) The timestep at which to freeze the State-Dependent Exploration (SDE) noise. If 0, SDE is never frozen.
- `--lr-milestones`: (str) Comma-separated list of timesteps for learning rate decay. Example: `60000,120000`.
- `--lr-factors`: (str) Comma-separated list of factors to multiply the learning rate at the milestones. Example: `0.33,0.5`.

#### TensorBoard Logging

You can enable TensorBoard logging by providing a log directory and an optional run name:

```bash
python backend/src/mg_rl_general/cli/train_sac.py \
    --model-json backend/examples/bh_bt_ri_complete_model_export.json \
    --config backend/src/mg_rl_general/examples/configs/emergent_ph_minimal.yaml \
    --timesteps 100000 \
    --n-envs 4 \
    --eval-freq 5000 \
    --tensorboard-log backend/logs/tb \
    --tb-log-name sac_general \
    --log-dir backend/logs/mg_rl_general \
    --model-path models/mg_rl_general_sac
```

Then launch TensorBoard in a separate terminal:

```bash
tensorboard --logdir backend/logs/tb --port 6006
```

Open `http://localhost:6006` to view training curves (e.g., `rollout/ep_rew_mean`, losses). Evaluation metrics recorded by the built-in evaluator are logged to the same run.

Example configs:
- `emergent_ph_minimal.yaml`: minimal emergent pH setup.
- `emergent_ph_parity.yaml`: parity with the restricted framework’s emergent pH.
- `emergent_ph_push15.yaml`: emergent pH with denser shaping aimed at pushing butyrate > 10–15 mM.

### `evaluate.py`

This script evaluates a trained model on the `GeneralMicrobiomeEnv`.

**Usage:**
```bash
python backend/src/mg_rl_general/cli/evaluate.py [OPTIONS]
```

**Arguments:**

- `--model-json`: (str) Path to the kinetic model JSON file.
- `--config`: (str) Path to the YAML/JSON configuration file for the environment.
- `--model-path`: (str) The path to the trained model `.zip` file.
- `--n-episodes`: (int, default: 10) The number of episodes to evaluate for.
- `--seed`: (int, default: 42) The random seed for reproducibility.

### `random_baseline.py`

This script runs a random agent on the `GeneralMicrobiomeEnv` to provide a baseline for comparison.

**Usage:**
```bash
python backend/src/mg_rl_general/cli/random_baseline.py [OPTIONS]
```

**Arguments:**

- `--model-json`: (str) Path to the kinetic model JSON file.
- `--config`: (str) Path to the YAML/JSON configuration file for the environment.
- `--episodes`: (int, default: 3) The number of episodes to run.
- `--seed`: (int, default: 42) The random seed for reproducibility.

### `plot_general_controller.py`

This script runs a trained SAC policy on the `GeneralMicrobiomeEnv` for a specified duration and produces publication-quality plots: target metabolite, pH, rewards, actions, and a kinetics replay (metabolites, pH, subpopulations, species).

Usage:
```bash
python backend/src/mg_rl_general/cli/plot_general_controller.py \
  --model-json backend/examples/bh_bt_ri_complete_model_export.json \
  --config backend/src/mg_rl_general/examples/configs/emergent_ph_minimal.yaml \
  --model-path models/mg_rl_general_sac.zip \
  --hours 500 \
  --det \
  --out backend/plots/mg_rl_general_run
```

Arguments:
- `--model-json`: (str) Path to the kinetic model JSON file.
- `--config`: (str) Path to YAML/JSON env config.
- `--model-path`: (str) Path to the trained model `.zip`.
- `--hours`: (float) Simulation duration in hours.
- `--det`: (flag) Use deterministic actions.
- `--out`: (str) Output directory for figures.
- `--baseline`: (float) Optional baseline target value overlay.
- `--norm-stats`: (str) Normalization stats JSON (auto-detected if omitted).
- `--no-stats`: (flag) Disable loading normalization stats.
- `--save-csv`: (flag) Export step-level series and kinetics CSVs.
- `--mets`: (str) Comma-separated metabolite names to plot in kinetics, or `all` to plot every metabolite.
- `--top-mets`: (int) When `--mets` not set, auto-include N metabolites with highest peak levels (default 4, always includes target).

Output directory:
- The script appends a timestamp to the provided `--out` name to avoid overwriting.
  - Example: `--out backend/plots/runA` creates `backend/plots/runA_20250911_123456`.
- If `--out` is omitted, a default directory `backend/plots/mg_rl_general_<model>_<timestamp>` is used.

## Key Concepts

This section explains some of the important concepts and components of the `mg_rl_general` framework.

### Observation Pipeline

The observation pipeline allows for preprocessing of the raw observation vector from the environment. It is defined by the `pipeline` key in the `observations` section of the configuration. The pipeline is a list of dictionaries, where each dictionary specifies a wrapper to apply. The wrappers are applied in the order they are listed.

The available wrappers are:

- **`normalize`**: Normalizes the observations.
  - **`method`**: The normalization method to use. The most common is `running_mean_var`, which normalizes the observations to have zero mean and unit variance, based on a running estimate.
  - **`clip`**: A tuple `[min, max]` to clip the normalized observations.
  - **`update_freq`**: The frequency (in timesteps) at which to update the running mean and variance.
  - **`epsilon`**: A small value to add to the variance to avoid division by zero.

- **`clip`**: Clips the observations to a specified range.
  - **`min`**: The minimum value.
  - **`max`**: The maximum value.

- **`history_stack`**: Stacks multiple historical observation frames into a single observation.
  - **`frames`**: The number of historical frames to stack.

**Examples:**
```yaml
observations:
  pipeline:
    - normalize: {method: running_mean_var, clip: [-5, 5]}
    - clip: {min: -10, max: 10}
    - history_stack: {frames: 4}
```
This pipeline first normalizes the observations, then clips them to the range `[-10, 10]`, and finally stacks 4 frames of history.

### Expression-based Rewards and Observations

A key feature of `mg_rl_general` is the ability to define reward terms and custom observation features using a simple expression language. This allows for rapid experimentation without needing to modify the environment code.

**Syntax:**

Expressions are strings that can include:
- **Metabolite concentrations**: `met['<metabolite_name>']` (e.g., `met['butyrate']`)
- **pH**: `pH`
- **Actions**: `action.<actuator_name>` (e.g., `action.q`, `action.temp`)
- **KPIs (Key Performance Indicators)**:
  - `kpi.sum_qdt`: The cumulative sum of `q * dt_hours`.
  - `kpi.sum_v`: The cumulative sum of `v`.
  - `kpi.pH_on`: The number of steps where pH control was active.
  - `kpi.t`: The current time in the episode (in hours).
  - `kpi.step`: The current step number in the episode.
- **Mathematical Operators**: `+`, `-`, `*`, `/`, `**` (power)
- **Functions**: `abs`, `min`, `max`, `clip`, `delta`, `rate`, `gate`
  - `delta(x)`: The change in `x` from the previous timestep.
  - `rate(x)`: The rate of change of `x` (`delta(x) / dt_hours`).
  - `gate(val, thr, out)`: Returns `out` if `val > thr`, otherwise 0.

**Example Reward Term:**
```yaml
rewards:
  terms:
    - {expr: "delta(met['butyrate'])", weight: 200.0, deadband: 0.02}
    - {expr: "action.q * dt_hours", weight: -0.02}
```
This reward function encourages the production of butyrate while penalizing the use of the feed `q`.

### Model Adapter

The `ModelAdapter` is an internal component that serves as a bridge between the `GeneralMicrobiomeEnv` and the underlying `mg_kinetic_model`. It is responsible for:
- Loading the kinetic model from a JSON file.
- Building the appropriate environment for the selected pH mode.
- Creating and integrating pulses (RL actions) into the simulation.
- Computing the emergent pH.

## Complete Example

Here is a complete example of a configuration file for training an agent in an emergent pH environment. The goal of the agent is to maximize the production of butyrate.

**`emergent_ph_train_example.yaml`**
```yaml
# Episode configuration
episode:
  horizon: 250
  dt_hours: 1.0
  training_mode: balanced

# Simulation configuration
simulation:
  min_steps_per_pulse: 10
  steps_per_hour_factor: 50

# Target for the reward function
target:
  type: metabolite
  name: butyrate
  use_delta: true

# Action space configuration
actions:
  pH_mode: emergent  # pH is not controlled by the agent
  actuators:
    - {name: q, type: continuous}
    - {name: v, type: continuous}
    - {name: stir, type: continuous}
    - {name: temp, type: continuous}
  bounds:
    q: [0.0, 0.5]
    v: [0.0, 0.2]
    stir: [0.0, 1.0]
    temp: [25.0, 45.0]

# Observation space configuration
observations:
  include:
    - met.all
    - pH.used
    - actuator_echo.all
    - met.delta.all
  pipeline:
    - normalize: {method: running_mean_var, clip: [-5, 5]}
    - clip: {min: -10, max: 10}

# Reward function configuration
rewards:
  error_reward: -1000.0
  terms:
    # Encourage production of the target metabolite (butyrate)
    - {expr: "delta(met['butyrate'])", weight: 200.0, deadband: 0.02}
    # Penalize control actions
    - {expr: "action.q * dt_hours", weight: -0.02}
    - {expr: "action.v", weight: -0.03}
    - {expr: "abs(action.temp - 37)", weight: -0.02}
    - {expr: "action.stir", weight: -0.01}
  terminal:
    # Bonus for high butyrate concentration at the end of the episode
    - {when: last_step, expr: "max(0, met['butyrate'] - 8.0)", weight: 15.0}
```

**Training Command:**
```bash
python backend/src/mg_rl_general/cli/train_sac.py \
    --model-json backend/examples/bh_bt_ri_complete_model_export.json \
    --config emergent_ph_train_example.yaml \
    --timesteps 100000 \
    --n-envs 4 \
    --eval-freq 5000 \
    --log-dir logs/emergent_ph_example \
    --model-path models/emergent_ph_example_sac
```

## Testing

The `mg_rl_general` package has a suite of tests to ensure its correctness and stability. To run the tests, you can use `pytest` from the root of the repository.

```bash
pytest backend/tests/
```

Some of the key test files for this package are:
- `backend/tests/test_mg_rl_general_env_smoke.py`: Smoke tests for the environment with different configurations.
- `backend/tests/test_mg_rl_general_cli_smoke.py`: Smoke tests for the CLI scripts.
- `backend/tests/test_mg_rl_general_pipeline.py`: Tests for the observation pipeline wrappers.
- `backend/tests/test_mg_rl_general_stability.py`: Slower tests that check for stability over longer rollouts.
- `backend/tests/test_mg_rl_general_expr_rewards.py`: Tests for the expression-based rewards.
Apply randomization to short episodes only (keep long episodes at JSON initial composition):

```yaml
init_randomization:
  enabled: true
  apply_to: short      # short|long|both
  metabolites:
    - {pattern: ".*", low: 0.0, high: 8.0}
```
