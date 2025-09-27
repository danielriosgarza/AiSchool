from __future__ import annotations

from typing import Dict, Tuple, Optional
import re

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import TopLevelConfig, compile_config
from .model_adapter import ModelAdapter, AdapterSettings
from .actions.schema import ActionSchema, ActionDecodeSettings
from .observation.builder import ObservationBuilder, ObservationSettings
from .rewards.engine import RewardEngine


class GeneralMicrobiomeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, model_json_path: str, config: TopLevelConfig):
        super().__init__()
        self.model_json_path = model_json_path
        self.cfg = compile_config(config, model_json_path)

        # Adapter
        self.adapter = ModelAdapter(
            model_json_path,
            AdapterSettings(
                dt_hours=self.cfg.episode.dt_hours,
                min_steps_per_pulse=self.cfg.simulation.min_steps_per_pulse,
                steps_per_hour_factor=self.cfg.simulation.steps_per_hour_factor,
                training_mode=self.cfg.episode.training_mode,
            ),
        )

        # Action schema
        bounds = self.cfg.actions["bounds"]
        self._pH_mode = str(self.cfg.actions.get("pH_mode", "switchable"))
        self.schema = ActionSchema(
            ActionDecodeSettings(
                bounds=bounds,
                pH_ctrl_threshold=float(bounds.get("pH_ctrl_threshold", 0.5)),
                smoothness_enabled=bool(self.cfg.actions.get("smoothness", {}).get("enabled", False)),
                max_delta=self.cfg.actions.get("smoothness", {}).get("max_delta", {}),
            )
        )

        # Observation builder with include-mapping
        include = set(self.cfg.observations.include)
        obs_settings = ObservationSettings(
            include_mets=("met.all" in include),
            include_pH_used=("pH.used" in include),
            include_deltas=("met.delta.all" in include),
            include_rates=("met.rate.all" in include),
            include_action_echo=("actuator_echo.all" in include),
            species_mode=self.cfg.observations.population.species.mode,
            include_species_deltas=("species.delta.live" in include),
            include_species_rates=("species.rate.live" in include),
        )
        self._obs_builder = ObservationBuilder(
            self.adapter.metabolome,
            self.adapter.microbiome,
            obs_settings,
        )
        # Compile observation expression features if any
        exprs = [s for s in self.cfg.observations.include if isinstance(s, str) and s.startswith("expr:")]
        exprs = [s.split("expr:", 1)[1].strip() for s in exprs]
        if exprs:
            self._obs_builder.set_feature_expressions(exprs)

        # Rewards engine (config-driven)
        self._rewards = RewardEngine(
            terms=self.cfg.rewards.get("terms", []), terminal=self.cfg.rewards.get("terminal", [])
        )

        # spaces
        self._setup_spaces()

        # episode state
        self.t: float = 0.0
        self.step_idx: int = 0
        self._prev_metabolites: Optional[np.ndarray] = None
        self._prev_state: Optional[np.ndarray] = None
        self._last_obs: Optional[np.ndarray] = None
        # KPIs
        self.sum_qdt: float = 0.0
        self.sum_v: float = 0.0
        self.pH_on: int = 0
        # Episode mix state
        self._episode_horizon: Optional[int] = None
        self._episode_type: str = "long"

    def _setup_spaces(self) -> None:
        include_pH = self._pH_mode in ("switchable", "controlled")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6 if include_pH else 4,), dtype=np.float32)
        # Lazy observation shape; will be confirmed at reset
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def _build_environment(self, decoded: Dict[str, float]):
        if self._pH_mode == "controlled":
            return self.adapter.build_env_controlled_pH(decoded["pH_set"], decoded["stir"], decoded["temp"])
        if self._pH_mode == "emergent":
            return self.adapter.build_env_emergent_pH(decoded["stir"], decoded["temp"])
        # switchable
        if int(decoded.get("pH_ctrl", 0)) == 1:
            return self.adapter.build_env_controlled_pH(decoded["pH_set"], decoded["stir"], decoded["temp"])
        return self.adapter.build_env_emergent_pH(decoded["stir"], decoded["temp"])

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Clear action decoder memory so new episodes start fresh
        if hasattr(self, "schema"):
            self.schema.reset()

        # rebuild adapter/model to restore initial state
        self.adapter = ModelAdapter(
            self.model_json_path,
            AdapterSettings(
                dt_hours=self.cfg.episode.dt_hours,
                min_steps_per_pulse=self.cfg.simulation.min_steps_per_pulse,
                steps_per_hour_factor=self.cfg.simulation.steps_per_hour_factor,
                training_mode=self.cfg.episode.training_mode,
            ),
        )
        self.t = 0.0
        self.step_idx = 0
        self.sum_qdt = 0.0
        self.sum_v = 0.0
        self.pH_on = 0

        # Episode mixing: choose horizon/type
        mix = self.cfg.episode.mix or {}
        ep_type = None
        if options and isinstance(options, dict):
            ep_type = options.get("episode_type")
        if ep_type not in ("short", "long"):
            if bool(mix.get("enabled", False)):
                p_short = float(mix.get("short_prob", 0.3))
                r = float(self.np_random.random()) if hasattr(self, "np_random") else float(np.random.default_rng().random())
                ep_type = "short" if r < p_short else "long"
            else:
                ep_type = "long"
        self._episode_type = ep_type
        if bool(mix.get("enabled", False)):
            self._episode_horizon = int(mix.get("short_horizon", 10) if ep_type == "short" else mix.get("long_horizon", 250))
        else:
            self._episode_horizon = int(self.cfg.episode.horizon)

        # Optional initialization randomization before reading state
        self._apply_init_randomization()

        state = self.adapter.reactor.get_states()
        mets = state[1 : 1 + self.adapter.n_metabolites()].copy()
        self._prev_metabolites = mets.copy()
        self._prev_state = state.copy()
        action_echo = np.zeros(6, dtype=np.float32)
        # respect mode at reset
        include_pH = self._pH_mode in ("switchable", "controlled")
        if self._pH_mode == "controlled":
            # use midpoint of bounds as default setpoint
            b = self.cfg.actions["bounds"]["pH_set"]
            pH_used = float(0.5 * (b[0] + b[1]))
        else:
            pH_used = self.adapter.compute_emergent_pH(mets)
        kpi = {
            "sum_qdt": self.sum_qdt,
            "sum_v": self.sum_v,
            "pH_on": self.pH_on,
            "t": self.t,
            "step": self.step_idx,
            "horizon": float(self._episode_horizon if self._episode_horizon is not None else self.cfg.episode.horizon),
        }
        obs, manifest = self._obs_builder.build(state, pH_used, action_echo, self._prev_metabolites, state, self.cfg.episode.dt_hours, kpi=kpi, extra_features={"delta_target": 0.0})
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs.size,), dtype=np.float32)
        self._last_obs = obs.copy()
        info = {
            "t": self.t,
            "step": self.step_idx,
            "episode_type": self._episode_type,
            "obs_manifest": manifest,
            "kpis_partial": {"sum_qdt": self.sum_qdt, "sum_v": self.sum_v, "pH_on": self.pH_on},
        }
        return obs, info

    def step(self, action: np.ndarray):
        include_pH = self._pH_mode in ("switchable", "controlled")
        decoded = self.schema.decode(action, include_pH=include_pH)
        env = self._build_environment(decoded)
        pulse = self.adapter.create_pulse(q=decoded["q"], v=decoded["v"], env=env)
        try:
            final_state = self.adapter.integrate_pulse(pulse, store_states=False)
        except Exception as exc:
            fallback_obs = self._last_obs.copy() if self._last_obs is not None else np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {"error": f"integration failed: {exc}", "step": self.step_idx}
            return fallback_obs, float(self.cfg.error_reward), True, False, info

        # numerical safety
        if not np.all(np.isfinite(final_state)):
            info = {"error": "non-finite state", "step": self.step_idx}
            return self._last_obs.copy(), float(self.cfg.error_reward), True, False, info

        # time & bookkeeping
        self.t += self.cfg.episode.dt_hours
        self.step_idx += 1
        # KPI accumulation
        self.sum_qdt += float(decoded["q"]) * float(self.cfg.episode.dt_hours)
        self.sum_v += float(decoded["v"])
        if include_pH and int(decoded.get("pH_ctrl", 0)) == 1:
            self.pH_on += 1

        # reward with richer feature namespace (selectors + action echo)
        mets_prev = self._prev_metabolites if self._prev_metabolites is not None else final_state[1 : 1 + self.adapter.n_metabolites()].copy()
        mets_now = final_state[1 : 1 + self.adapter.n_metabolites()].copy()
        # current episode horizon for parity with restricted env scaling
        horizon = int(self._episode_horizon if self._episode_horizon is not None else self.cfg.episode.horizon)
        # Build features for rewards (config-driven)
        feat = {
            "dt_hours": float(self.cfg.episode.dt_hours),
            # action accessors (both dotted and underscored for compatibility)
            "action.q": float(decoded["q"]),
            "action_v": float(decoded["v"]),
            "action.v": float(decoded["v"]),
            "action_q": float(decoded["q"]),
            "action.pH_ctrl": float(decoded.get("pH_ctrl", 0)),
            "action_pH_ctrl": float(decoded.get("pH_ctrl", 0)),
            "action.pH_set": float(decoded.get("pH_set", 0.0)),
            "action_pH_set": float(decoded.get("pH_set", 0.0)),
            "action.stir": float(decoded["stir"]),
            "action_stir": float(decoded["stir"]),
            "action.temp": float(decoded["temp"]),
            "action_temp": float(decoded["temp"]),
            # KPI selectors
            "kpi.sum_qdt": float(self.sum_qdt),
            "kpi.sum_v": float(self.sum_v),
            "kpi.pH_on": float(self.pH_on),
            "kpi.t": float(self.t),
            "kpi.step": float(self.step_idx),
        }
        feat["delta_target"] = float(feat.get("delta_target", 0.0))
        feat["kpi.horizon"] = float(horizon)
        # Add metabolite selectors by name: met['name']
        for i, name in enumerate(self.adapter.metabolome.metabolites):
            feat[f"met['{name}']"] = float(mets_now[i])
        species_total_live = 0.0
        try:
            species_names = list(self.adapter.microbiome.bacteria.keys())
            live = self._obs_builder.species_live_signals(final_state)
            n_species = min(len(species_names), live.size // 2)
            live_counts = live[:n_species]
            for idx, name in enumerate(species_names[:n_species]):
                count = float(live_counts[idx])
                feat[f"species.live_count['{name}']"] = count
                species_total_live += count
            feat['species.live_count.total'] = species_total_live
        except Exception:
            feat['species.live_count.total'] = species_total_live
        prev_feat = {k: v for k, v in feat.items()}
        for i, name in enumerate(self.adapter.metabolome.metabolites):
            prev_feat[f"met['{name}']"] = float(mets_prev[i])
        # Previous species live counts (for delta/species)
        try:
            if self._prev_state is not None:
                prev_live = self._obs_builder.species_live_signals(self._prev_state)
                species_names = list(self.adapter.microbiome.bacteria.keys())
                n_species = min(len(species_names), prev_live.size // 2)
                prev_counts = prev_live[:n_species]
                species_total_live_prev = 0.0
                for idx, name in enumerate(species_names[:n_species]):
                    c = float(prev_counts[idx])
                    prev_feat[f"species.live_count['{name}']"] = c
                    species_total_live_prev += c
                prev_feat['species.live_count.total'] = species_total_live_prev
        except Exception:
            pass
        # also keep legacy delta_target for backward-compat configs, but only if target is explicitly set
        try:
            tgt = getattr(self.cfg, "target", None) or {}
            tname = tgt.get("name", None)
            if isinstance(tname, str) and len(tname) > 0:
                target_idx = int(self.adapter.get_metabolite_index(str(tname)))
                feat["delta_target"] = float(mets_now[target_idx] - mets_prev[target_idx])
        except Exception:
            pass
        # Episode-type meta for reward gating
        feat["__episode_type"] = self._episode_type
        
        # DEBUG: Print features to check horizon
        if self.step_idx == 1:  # Only print on first step
            print(f"DEBUG: kpi.horizon in features: {feat.get('kpi.horizon', 'NOT_FOUND')}")
            print(f"DEBUG: horizon variable: {horizon}")
            print(f"DEBUG: episode_type: {self._episode_type}")
        
        step_reward, step_bd = self._rewards.compute_step(feat, prev_features=prev_feat, dt_hours=self.cfg.episode.dt_hours)
        reward = step_reward
        breakdown = step_bd

        # pH used
        pH_used = decoded["pH_set"] if (include_pH and int(decoded.get("pH_ctrl", 0)) == 1) else self.adapter.compute_emergent_pH(mets_now)
        action_echo = np.array([decoded[k] for k in ("q", "v", "pH_ctrl", "pH_set", "stir", "temp")], dtype=np.float32)
        kpi = {"sum_qdt": self.sum_qdt, "sum_v": self.sum_v, "pH_on": self.pH_on, "t": self.t, "step": self.step_idx, "horizon": float(horizon)}
        obs, manifest = self._obs_builder.build(final_state, pH_used, action_echo, mets_prev, final_state, self.cfg.episode.dt_hours, kpi=kpi, extra_features=feat)

        # termination by horizon (per-episode horizon)
        truncated = self.step_idx >= horizon
        terminated = False

        # terminal rewards on last step
        if truncated:
            terminal_r, term_bd = self._rewards.compute_terminal(feat, prev_features=prev_feat, dt_hours=self.cfg.episode.dt_hours)
            reward = float(reward + terminal_r)
            breakdown.update(term_bd)

        self._prev_metabolites = mets_now.copy()
        # keep last full state for next-step deltas
        try:
            self._prev_state = final_state.copy()
        except Exception:
            self._prev_state = None
        self._last_obs = obs.copy()

        # For diagnostics/parity: include target concentration if target explicitly provided
        _B_now = float('nan')
        try:
            tgt = getattr(self.cfg, "target", None) or {}
            tname = tgt.get("name", None)
            if isinstance(tname, str) and len(tname) > 0:
                _tidx = int(self.adapter.get_metabolite_index(str(tname)))
                _B_now = float(mets_now[_tidx])
        except Exception:
            _B_now = float('nan')
        info = {
            "t": self.t,
            "step": self.step_idx,
            "episode_type": self._episode_type,
            "obs_manifest": manifest,
            "reward_breakdown": breakdown,
            "kpis_partial": {"sum_qdt": self.sum_qdt, "sum_v": self.sum_v, "pH_on": self.pH_on},
            "target_concentration": _B_now,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    

    def _apply_init_randomization(self) -> None:
        """Randomize initial metabolite concentrations based on config patterns.

        Safe no-op on errors to prioritize robustness.
        """
        try:
            init = getattr(self.cfg, "init_randomization", None)
            if not init or not bool(init.get("enabled", False)):
                return
            # Respect episode type targeting
            apply_to = str(init.get("apply_to", "both")).lower()
            if apply_to not in ("short", "long", "both"):
                apply_to = "both"
            if apply_to != "both" and getattr(self, "_episode_type", "long") != apply_to:
                return
            rng = getattr(self, "np_random", None) or np.random.default_rng()
            rules = init.get("metabolites", []) or []
            if rules:
                updates: Dict[str, float] = {}
                names = list(self.adapter.metabolome.metabolites)
                for rule in rules:
                    if not isinstance(rule, dict):
                        continue
                    pat = str(rule.get("pattern", ".*"))
                    lo = float(rule.get("low", 0.0))
                    hi = float(rule.get("high", lo))
                    if hi < lo:
                        lo, hi = hi, lo
                    cre = re.compile(pat)
                    for name in names:
                        if cre.match(name):
                            updates[name] = float(lo if hi == lo else rng.uniform(lo, hi))
                if updates:
                    self.adapter.metabolome.update(updates)

            sub_rules = init.get("subpopulations", []) or []
            if sub_rules:
                for rule in sub_rules:
                    if not isinstance(rule, dict):
                        continue
                    lo = float(rule.get("low", 0.0))
                    hi = float(rule.get("high", lo))
                    if hi < lo:
                        lo, hi = hi, lo
                    state_filter = str(rule.get("state", "active")).lower()
                    species_filter = rule.get("species")
                    name_pattern = rule.get("pattern")
                    name_regex = re.compile(str(name_pattern)) if isinstance(name_pattern, str) and name_pattern else None
                    for species_name, bacteria in self.adapter.microbiome.bacteria.items():
                        if species_filter and str(species_filter).lower() != str(species_name).lower():
                            continue
                        for sub_name, sub in bacteria.subpopulations.items():
                            if state_filter not in ("*", "all") and str(sub.state).lower() != state_filter:
                                continue
                            if name_regex and not name_regex.match(sub_name):
                                continue
                            new_count = float(lo if hi == lo else rng.uniform(lo, hi))
                            sub.update(new_count)
        except Exception:
            return

