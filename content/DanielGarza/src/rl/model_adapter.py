from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from kinetic_model.model_from_json import ModelFromJson
from kinetic_model.reactor import Reactor, Pulse
from kinetic_model.environment import Environment, pH, Stirring, Temperature
from kinetic_model.metabolome import Metabolome
from kinetic_model.metabolite import Metabolite
from kinetic_model.microbiome import Microbiome


@dataclass
class AdapterSettings:
    dt_hours: float
    min_steps_per_pulse: int
    steps_per_hour_factor: int
    volume_liters: float = 1.0
    training_mode: str = "fast"  # fast|balanced|accurate


class ModelAdapter:
    """
    Bridges mg_kinetic_model with the general RL environment.
    """

    def __init__(self, model_json_path: str, settings: AdapterSettings):
        self.model_json_path = model_json_path
        self.settings = settings

        self.model = ModelFromJson(model_json_path)
        self.metabolome: Metabolome = self.model.metabolome
        self.microbiome: Microbiome = self.model.microbiome
        self.environment: Environment = self.model.environment

        self.reactor: Reactor = Reactor(
            microbiome=self.microbiome,
            metabolome=self.metabolome,
            pulses=[],
            volume=self.settings.volume_liters,
        )

        # apply simulation mode presets
        self.set_simulation_mode(self.settings.training_mode)

        # cache metabolite lookup
        self._met_index = {name: i for i, name in enumerate(self.metabolome.metabolites)}

        # cache emergent pH configuration from model
        self._emergent_intercept = self.environment.pH._get_current_intercept() if hasattr(self.environment.pH, "_get_current_intercept") else self.environment.pH.pH
        self._emergent_weights = self.environment.pH._get_current_met_dictionary() if hasattr(self.environment.pH, "_get_current_met_dictionary") else {}

        # Build default feed metabolome from initial medium composition (parity with restricted env)
        feed_metabolites = []
        for met_name in self.metabolome.metabolites:
            met = self.metabolome._metabolite_dict[met_name]
            try:
                formula = dict(met.formula)
            except Exception:
                formula = {}
            feed_metabolites.append(
                Metabolite(
                    name=met.name,
                    concentration=met.concentration,
                    formula=formula,
                    color=met.color,
                    description=getattr(met, "description", ""),
                )
            )
        if not feed_metabolites:
            raise ValueError("Loaded model must define at least one metabolite to build feed composition")
        self.feed_metabolome = Metabolome(metabolites=feed_metabolites)

    # --- Info helpers ---
    def get_metabolite_index(self, name: str) -> int:
        if name not in self._met_index:
            raise KeyError(f"unknown metabolite: {name}")
        return self._met_index[name]

    def n_metabolites(self) -> int:
        return self.metabolome.nmets

    def species_names(self):
        return list(self.microbiome.bacteria.keys())

    # --- Environment builders ---
    def build_env_controlled_pH(self, setpoint: float, stir: float, temp_c: float) -> Environment:
        env = Environment(
            pH(self.metabolome, intercept=setpoint, met_dictionary={}),
            Stirring(rate=float(stir), base_std=0.02),
            Temperature(float(temp_c)),
        )
        return env

    def build_env_emergent_pH(self, stir: float, temp_c: float) -> Environment:
        # Use the pH object from the loaded model environment for exact parity with restricted framework
        env = Environment(
            self.environment.pH,
            Stirring(rate=float(stir), base_std=0.02),
            Temperature(float(temp_c)),
        )
        return env

    def compute_emergent_pH(self, concentrations: np.ndarray) -> float:
        env = self.build_env_emergent_pH(stir=1.0, temp_c=37.0)
        return env.pH.compute_pH(concentrations)

    # --- Pulse and integration ---
    def _compute_n_steps(self) -> int:
        return max(self.settings.min_steps_per_pulse, int(self.settings.steps_per_hour_factor * self.settings.dt_hours))

    def create_pulse(self, q: float, v: float, env: Environment) -> Pulse:
        t = 0.0 if getattr(self, "t", None) is None else float(self.t)
        pulse = Pulse(
            t_start=t,
            t_end=t + self.settings.dt_hours,
            n_steps=self._compute_n_steps(),
            vin=float(v),
            vout=float(v),
            qin=float(q),
            qout=float(q),
            # Provide feed composition for both instant and continuous flows
            instant_feed_metabolome=self.feed_metabolome,
            continuous_feed_metabolome=self.feed_metabolome,
            environment=env,
        )
        return pulse

    def integrate_pulse(self, pulse: Pulse, store_states: bool = False) -> np.ndarray:
        final_state = self.reactor.integrate_pulse(pulse, store_states=store_states)
        # advance adapter time
        self.t = float(pulse.t_end)
        return final_state

    # --- Solver mode ---
    def set_simulation_mode(self, mode: str) -> None:
        mode = str(mode)
        if mode == "fast":
            self.reactor.set_fast_simulation_mode()
        elif mode == "balanced":
            self.reactor.set_balanced_simulation_mode()
        elif mode == "accurate":
            # default tolerances (no reductions)
            pass
        else:
            # ignore unknown mode silently for robustness
            pass


