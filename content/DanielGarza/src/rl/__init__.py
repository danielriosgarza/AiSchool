"""
mg_rl_general
~~~~~~~~~~~~~~

A config-driven, model-agnostic RL package for controlling microbiome kinetic
models via `mg_kinetic_model`. This package is independent of the existing
`mg_rl` and follows the general plan documented in
`backend/general_RL_implementation_plan_UPDATED.md`.
"""

__version__ = "0.1.0"

from .env import GeneralMicrobiomeEnv

__all__ = ["GeneralMicrobiomeEnv"]


