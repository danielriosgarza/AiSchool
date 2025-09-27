from __future__ import annotations

from typing import Dict, List, Optional, Literal, Tuple, Any

from pydantic import BaseModel, Field, ConfigDict, ValidationError, model_validator


class EpisodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizon: int = Field(250, ge=1)
    dt_hours: float = Field(1.0, gt=0)
    training_mode: Literal["fast", "balanced", "accurate"] = "balanced"
    randomize_horizon: bool = False
    horizon_range: Tuple[int, int] = (100, 250)
    # Optional short/long mixing (if enabled, overrides fixed horizon)
    mix: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": False,
            "short_horizon": 10,
            "long_horizon": 250,
            "short_prob": 0.3,
        }
    )


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_steps_per_pulse: int = Field(10, ge=1)
    steps_per_hour_factor: int = Field(50, ge=1)


class ActionBounds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: Tuple[float, float] = (0.0, 0.5)
    v: Tuple[float, float] = (0.0, 0.2)
    pH_set: Tuple[float, float] = (5.8, 7.8)
    stir: Tuple[float, float] = (0.0, 1.0)
    temp: Tuple[float, float] = (25.0, 45.0)
    pH_ctrl_threshold: float = Field(0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_bounds(cls, v: "ActionBounds") -> "ActionBounds":
        def _chk(pair: Tuple[float, float], name: str) -> None:
            if pair[0] >= pair[1]:
                raise ValueError(f"{name} must satisfy low < high, got {pair}")

        _chk(v.q, "q")
        _chk(v.v, "v")
        _chk(v.pH_set, "pH_set")
        _chk(v.stir, "stir")
        _chk(v.temp, "temp")
        return v


class RewardWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(200.0, ge=0)
    beta_q: float = Field(0.02, ge=0)
    beta_v: float = Field(0.03, ge=0)
    beta_ctrl: float = Field(0.01, ge=0)
    beta_temp: float = Field(0.02, ge=0)
    beta_stir: float = Field(0.01, ge=0)
    # Optional smoothness terms (kept for forward compatibility)
    beta_dq: float = Field(0.0, ge=0)
    beta_dv: float = Field(0.0, ge=0)
    beta_dtemp: float = Field(0.0, ge=0)
    beta_dstir: float = Field(0.0, ge=0)


class ObservationPopulationSpecies(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["off", "live", "aggregates", "summary"] = "live"
    signals: List[Literal["live_count", "live_share"]] = Field(
        default_factory=lambda: ["live_count", "live_share"]
    )
    normalization: Literal["sum1", "log1p", "clr"] = "sum1"
    top_k: Optional[int] = None


class ObservationPopulationSubpopsDense(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: Literal["model"] = "model"
    signals: List[str] = Field(default_factory=lambda: ["live_count"])
    pad_to: Optional[int] = None


class ObservationPopulationSubpopsTokens(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: List[str] = Field(
        default_factory=lambda: ["live_count", "growth", "state_onehot", "pH_opt", "temp_opt"]
    )
    pool: Literal["mean", "max", "attention"] = "attention"
    history_frames: int = Field(1, ge=1)
    top_k_per_species: Optional[int] = None
    pad_to: Optional[int] = None


class ObservationPopulationSubpops(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["off", "dense", "tokens"] = "tokens"
    dense: ObservationPopulationSubpopsDense = ObservationPopulationSubpopsDense()
    tokens: ObservationPopulationSubpopsTokens = ObservationPopulationSubpopsTokens()


class ObservationPopulation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    species: ObservationPopulationSpecies = ObservationPopulationSpecies()
    subpopulations: ObservationPopulationSubpops = ObservationPopulationSubpops()


class ObservationsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include: List[str] = Field(
        default_factory=lambda: [
            "met.all",
            "pH.used",
            "actuator_echo.all",
            "met.delta.all",
            "met.rate.all",
        ]
    )
    population: ObservationPopulation = ObservationPopulation()
    pipeline: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {"normalize": {"method": "running_mean_var"}},
            {"clip": {"min": -10, "max": 10}},
        ]
    )


class TopLevelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode: EpisodeConfig = EpisodeConfig()
    simulation: SimulationConfig = SimulationConfig()
    seed: Optional[int] = None
    target: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "metabolite", "name": "butyrate", "use_delta": True}
    )
    actions: Dict[str, Any] = Field(
        default_factory=lambda: {
            "pH_mode": "switchable",  # controlled|emergent|switchable
            "actuators": [
                {"name": "q", "type": "continuous"},
                {"name": "v", "type": "continuous"},
                {"name": "pH_ctrl", "type": "binary"},
                {"name": "pH_set", "type": "continuous"},
                {"name": "stir", "type": "continuous"},
                {"name": "temp", "type": "continuous"},
            ],
            "bounds": ActionBounds().model_dump(),
            "smoothness": {"enabled": False, "max_delta": {}},
        }
    )
    observations: ObservationsConfig = ObservationsConfig()
    rewards: Dict[str, Any] = Field(
        default_factory=lambda: {
            "error_reward": -1000.0,
            "terms": [
                {"expr": "delta_target", "weight": 200.0, "deadband": 0.02},
                {"expr": "action_q * dt_hours", "weight": -0.02},
                {"expr": "action_v", "weight": -0.03},
                {"expr": "abs(action_temp - 37)", "weight": -0.02},
                {"expr": "action_stir", "weight": -0.01},
            ],
            "terminal": [
                {"when": "last_step", "expr": "max(0, delta_target)", "weight": 15.0}
            ],
        }
    )
    init_randomization: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": False,
            "apply_to": "both",  # short|long|both
            "metabolites": [{"pattern": ".*", "low": 0.0, "high": 8.0}],
            "subpopulations": [{"state": "active", "low": 0.0, "high": 8.0}],
        }
    )
    error_reward: float = -1000.0

    # Cross-field invariants and mode-dependent constraints
    @model_validator(mode="after")
    def _cross_checks(cls, v: "TopLevelConfig") -> "TopLevelConfig":
        # Actions bounds sanity
        try:
            ActionBounds(**v.actions.get("bounds", {}))
        except ValidationError as e:
            raise ValueError(f"actions.bounds invalid: {e}")

        # Ensure observations include at least one feature set
        if not v.observations.include:
            raise ValueError("observations.include must not be empty")

        # Target sanity
        tgt = v.target or {}
        if tgt.get("type") not in {"metabolite", "biomass", "composite"}:
            raise ValueError("target.type must be metabolite|biomass|composite")
        if tgt.get("type") in {"metabolite", "biomass"} and not isinstance(tgt.get("name"), str):
            raise ValueError("target.name must be a string for metabolite/biomass")

        return v


def compile_config(cfg: TopLevelConfig, model_json_path: str) -> TopLevelConfig:
    """
    Compile-time checks that depend on the loaded model (selectors, expressions, modes).
    This does not mutate cfg; it validates and raises on errors.
    """
    # Deferred import to avoid hard coupling
    from kinetic_model.model_from_json import ModelFromJson

    model = ModelFromJson(model_json_path)

    # Validate selector-like strings in observations.include and rewards terms
    # (placeholder: actual AST and selector resolution will be implemented in M2)
    _known = set(model.metabolome.metabolites)
    # Simple heuristic validations for now
    for item in cfg.observations.include:
        if not isinstance(item, str):
            raise ValueError(f"observations.include entries must be strings, got {item}")

    for term in cfg.rewards.get("terms", []):
        if not isinstance(term, dict) or "expr" not in term:
            raise ValueError("each rewards.terms item must be a dict with 'expr'")
        if "when" in term and term["when"] not in ("short", "long", "always"):
            raise ValueError("rewards.terms.when must be 'short'|'long'|'always'")

    # Validate init_randomization.apply_to
    init = cfg.init_randomization or {}
    if init:
        apply_to = str(init.get("apply_to", "both")).lower()
        if apply_to not in ("short", "long", "both"):
            raise ValueError("init_randomization.apply_to must be 'short'|'long'|'both'")

    return cfg


