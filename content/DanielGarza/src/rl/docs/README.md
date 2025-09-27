mg_rl_general – General RL Framework for Microbiome Control
===========================================================

This package provides a config-driven, model-agnostic RL environment and tooling built on top of `mg_kinetic_model`. It supports emergent and controlled pH, safe expression-based observations/rewards, normalization/history wrappers, and SAC training/evaluation CLIs.

Key entry points
- Env: `mg_rl_general.env.GeneralMicrobiomeEnv`
- Config: `mg_rl_general.config.TopLevelConfig` (YAML/JSON loadable)
- CLIs: `mg_rl_general/cli/train_sac.py`, `evaluate.py`, `random_baseline.py`
- Examples: `mg_rl_general/examples/configs/`

Quickstart
- Train (emergent pH, SAC):
  python backend/src/mg_rl_general/cli/train_sac.py --model-json backend/examples/bh_bt_ri_complete_model_export.json --config backend/src/mg_rl_general/examples/configs/emergent_ph_train.yaml --timesteps 100000 --n-envs 4 --eval-freq 5000 --log-dir logs/mg_rl_general --model-path models/mg_rl_general_sac
- Evaluate (accurate mode):
  python backend/src/mg_rl_general/cli/evaluate.py --model-json backend/examples/bh_bt_ri_complete_model_export.json --config backend/src/mg_rl_general/examples/configs/emergent_ph_train.yaml --model-path models/mg_rl_general_sac.zip
- Random baseline:
  python backend/src/mg_rl_general/cli/random_baseline.py --model-json backend/examples/bh_bt_ri_complete_model_export.json --config backend/src/mg_rl_general/examples/configs/emergent_ph_train.yaml --episodes 3

Artifacts
- Models/checkpoints: `models/`
- Normalization stats JSON: alongside model (`*_normalization_stats.json`)
- Logs: `logs/`

Restricted Parity (Emergent pH)
- Map restricted trainer to general framework:
  - Action space: `actions.pH_mode: emergent`, actuators `[q, v, stir, temp]`.
  - Episode: `episode.dt_hours`, `episode.horizon`, `episode.training_mode`.
  - Simulation: `simulation.min_steps_per_pulse`, `simulation.steps_per_hour_factor`.
  - Observations: `met.all`, `pH.used`, `actuator_echo.all`, `met.delta.all`, `met.rate.all`.
  - Rewards: use expressions for target delta and control costs.
  - Normalization: pipeline with `normalize` and optional `clip`; freeze later via CLI flag.
  - Evaluation: CLIs run eval env in accurate mode.
- Example config: `backend/src/mg_rl_general/examples/configs/emergent_ph_train.yaml`.
- Optional schedules (capability via flags):
  - Entropy target: `--entropy-target-at 60000:-4,120000:-6`
  - SDE freeze: `--sde-freeze-at 60000`
  - Stepwise LR: `--lr-milestones 60000,120000 --lr-factors 0.33,0.5`
  - Freeze normalization: `--freeze-norm-steps 25000`
- Example command:
  python backend/src/mg_rl_general/cli/train_sac.py --model-json backend/examples/bh_bt_ri_complete_model_export.json --config backend/src/mg_rl_general/examples/configs/emergent_ph_train.yaml --timesteps 100000 --n-envs 4 --eval-freq 5000 --entropy-target-at 60000:-4,120000:-6 --sde-freeze-at 60000 --lr-milestones 60000,120000 --lr-factors 0.33,0.5 --freeze-norm-steps 25000 --log-dir logs/sac_emergent_ph_general --model-path models/sac_emergent_ph_general

Observation Pipeline
- Applied automatically by CLIs from `observations.pipeline`:
  - `normalize: {method: running_mean_var, clip: [-5,5], update_freq, epsilon}`
  - `clip: {min, max}`
  - `history_stack: {frames}`
- Example YAML:
  observations:
    include: [met.all, pH.used, actuator_echo.all, met.delta.all, met.rate.all]
    pipeline:
      - normalize: {method: running_mean_var, clip: [-5, 5]}
      - clip: {min: -10, max: 10}
      - history_stack: {frames: 4}

Expressions (Obs + Rewards)
- Features: `met['name']`, `pH`, `action.q|v|pH_ctrl|pH_set|stir|temp`
- KPIs: `kpi.sum_qdt`, `kpi.sum_v`, `kpi.pH_on`, `kpi.t`, `kpi.step`
- Ops: +, -, *, /, **, abs, min, max, clip, delta(x), rate(x), gate(val, thr, out)
- Reward example:
  rewards:
    terms:
      - {expr: "delta(met['butyrate'])", weight: 200.0, deadband: 0.02}
      - {expr: "action.q * dt_hours", weight: -0.02}
      - {expr: "action.v", weight: -0.03}
      - {expr: "abs(action.temp - 37)", weight: -0.02}
      - {expr: "action.stir", weight: -0.01}
    terminal:
      - {when: last_step, expr: "max(0, met['butyrate'] - 8.0)", weight: 15.0}

Tips
- Use `--seed` on train/eval for reproducibility.
- For faster tests reduce `min_steps_per_pulse`, `steps_per_hour_factor`, `dt_hours`.

Tests
- CLI smoke: `backend/tests/test_mg_rl_general_cli_smoke.py`
- Stability: `backend/tests/test_mg_rl_general_stability.py` (slow)
- Pipeline: `backend/tests/test_mg_rl_general_pipeline.py`

