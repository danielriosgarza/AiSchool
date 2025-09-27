General RL Short Tutorial (Emergent pH, SAC)
===========================================

This tutorial walks through training and evaluating a Soft Actor–Critic (SAC) agent using the general framework in emergent pH mode. The provided configuration mirrors an emergent‑pH setup: action space excludes pH control, reward emphasizes butyrate level above a threshold with per‑step control costs, and the episode horizon is long‑only.

1) Create the Parity Config
Save the following as `backend/src/mg_rl_general/examples/configs/emergent_ph_parity.yaml`.

```yaml
# Episode configuration (long-only)
episode:
  horizon: 250          # long episodes
  dt_hours: 1.0
  training_mode: fast   # fast mode for training

# Simulation (integration steps)
simulation:
  min_steps_per_pulse: 10
  steps_per_hour_factor: 50

# Action space (emergent pH; 4D actions)
actions:
  pH_mode: emergent
  actuators:
    - {name: q,    type: continuous}
    - {name: v,    type: continuous}
    - {name: stir, type: continuous}
    - {name: temp, type: continuous}
  bounds:
    q:    [0.0, 0.2]
    v:    [0.0, 0.1]
    stir: [0.0, 1.0]
    temp: [25.0, 45.0]

# Observations and pipeline
observations:
  include:
    - met.all
    - pH.used
    - actuator_echo.all
    - met.delta.all
    - met.rate.all
  pipeline:
    - normalize: {method: running_mean_var, clip: [-5, 5]}
    - clip: {min: -10, max: 10}

# Reward (per-step level shaping + control costs; terminal level + progressive bonus)
# Notation: B = met['butyrate']
rewards:
  error_reward: -1000.0
  terms:
    # Per-step level shaping above threshold (weight_step = 0.75)
    # max(0, B - 8.0) ** 1.0
    - {expr: "(clip(met['butyrate'] - 8.0, 0, 1e9) ** 1.0)", weight: 0.75}

    # Control costs (betas):
    - {expr: "action.q * dt_hours",          weight: -0.015}
    - {expr: "action.v",                      weight: -0.025}
    - {expr: "abs(action.temp - 37)",         weight: -0.015}
    - {expr: "action.stir",                   weight: -0.008}

  terminal:
    # Terminal level bonus: weight_terminal * max(0, B - 8.0) ** 1.0 (weight_terminal = 15.0)
    - {when: last_step, expr: "(clip(met['butyrate'] - 8.0, 0, 1e9) ** 1.0)", weight: 15.0}

    # Progressive terminal bonus (long episodes): 25 * (p^2) * B,
    # where p = clip((B - 1.0) / 9.0, 0, 1)
    - {when: last_step, expr: "(clip((met['butyrate'] - 1.0) / 9.0, 0, 1.0) ** 2) * met['butyrate']", weight: 25.0}
```

Notes
- Action space is `[q, v, stir, temp]` (no pH control); pH is computed from metabolites.
- Horizon is long‑only (250). Episode‑mixing (short/long) is not used here.
- Per‑step production (delta term) is intentionally omitted; long‑episode shaping is level‑based plus terminal bonuses.

2) Train with SAC
Run SAC training with 4 parallel environments and periodic evaluation.

```bash
python backend/src/mg_rl_general/cli/train_sac.py \
  --model-json backend/examples/bh_bt_ri_complete_model_export.json \
  --config backend/src/mg_rl_general/examples/configs/emergent_ph_parity.yaml \
  --timesteps 100000 \
  --n-envs 4 \
  --eval-freq 5000 \
  --log-dir logs/sac_emergent_ph_parity \
  --model-path models/sac_emergent_ph_parity
```

Artifacts
- Trained model: `models/sac_emergent_ph_parity.zip`
- Normalization stats: `models/sac_emergent_ph_parity_normalization_stats.json`
- Logs: `logs/sac_emergent_ph_parity/`

3) Evaluate the Trained Model
Run deterministic evaluation using the same config.

```bash
python backend/src/mg_rl_general/cli/evaluate.py \
  --model-json backend/examples/bh_bt_ri_complete_model_export.json \
  --config backend/src/mg_rl_general/examples/configs/emergent_ph_parity.yaml \
  --model-path models/sac_emergent_ph_parity.zip
```

4) (Optional) Plot Controller Trajectories
Use the plotting helper to generate summary figures over a specified duration.

```bash
python backend/src/mg_rl_general/scripts/plot_emergent_ph_controller.py \
  --model models/sac_emergent_ph_parity.zip \
  --model-json backend/examples/bh_bt_ri_complete_model_export.json \
  --norm-stats models/sac_emergent_ph_parity_normalization_stats.json \
  --hours 500 \
  --det \
  --out plots/emergent_ph_parity
```

Troubleshooting Tips
- If training appears slow, reduce `simulation.min_steps_per_pulse`, `simulation.steps_per_hour_factor`, or `episode.dt_hours` temporarily for testing.
- Ensure the model JSON path is correct: `backend/examples/bh_bt_ri_complete_model_export.json`.
- Normalization is enabled via the observation pipeline and applied automatically by the CLIs.

Optional: Use Episode Mixing (Short + Long)
------------------------------------------
If you want to mix short and long episodes, save this config as `backend/src/mg_rl_general/examples/configs/emergent_ph_parity_mix.yaml` and train with `--config emergent_ph_parity_mix.yaml`.

```yaml
episode:
  mix: {enabled: true, short_horizon: 10, long_horizon: 250, short_prob: 0.3}
  dt_hours: 1.0
  training_mode: fast

simulation:
  min_steps_per_pulse: 10
  steps_per_hour_factor: 50

actions:
  pH_mode: emergent
  actuators:
    - {name: q,    type: continuous}
    - {name: v,    type: continuous}
    - {name: stir, type: continuous}
    - {name: temp, type: continuous}
  bounds:
    q:    [0.0, 0.2]
    v:    [0.0, 0.1]
    stir: [0.0, 1.0]
    temp: [25.0, 45.0]

observations:
  include: [met.all, pH.used, actuator_echo.all, met.delta.all, met.rate.all]
  pipeline:
    - normalize: {method: running_mean_var, clip: [-5, 5]}
    - clip: {min: -10, max: 10}

rewards:
  error_reward: -1000.0
  terms:
    # Short-only production: alpha * g(B) * delta(B) with delta deadband 0.02
    # g(B) = clip((B - 2.0) / (8.0 - 2.0), 0, 1)
    # deadband on delta only: gate(abs(delta), 0.02, delta)
    - {expr: "clip((met['butyrate'] - 2.0) / 6.0, 0, 1) * gate(abs(delta(met['butyrate'])), 0.02, delta(met['butyrate']))", weight: 200.0, when: short}

    # Per-step level shaping (always)
    - {expr: "(clip(met['butyrate'] - 8.0, 0, 1e9) ** 1.0)", weight: 0.75}

    # Control costs (always)
    - {expr: "action.q * dt_hours",          weight: -0.015}
    - {expr: "action.v",                      weight: -0.025}
    - {expr: "abs(action.temp - 37)",         weight: -0.015}
    - {expr: "action.stir",                   weight: -0.008}

  terminal:
    # Terminal level bonus (always)
    - {when: last_step, expr: "(clip(met['butyrate'] - 8.0, 0, 1e9) ** 1.0)", weight: 15.0}

    # Progressive terminal bonus (long-only)
    - {when: last_step, expr: "(clip((met['butyrate'] - 1.0) / 9.0, 0, 1.0) ** 2) * met['butyrate']", weight: 25.0, when: long}
```
