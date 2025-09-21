# -*- coding: utf-8 -*-
"""
kinetic_model.temperature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines :class:`~kinetic_model.temperature.Temperature`, which manages
temperature (°C) in the kinetic model system.

Example
-------
>>> from kinetic_model.temperature import Temperature
>>> temp = Temperature(37.0)
>>> temp.temperature
37.0
>>> temp.set_temperature(42.0)
>>> temp.temperature
42.0
>>> temp.to_json()            # '{"temperature": 42.0, ...}'
"""

from __future__ import annotations

import json
import numbers
from typing import Any, Dict

# Conservative biological bounds (°C)
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 100.0


class Temperature:
    """
    Represents temperature with internal state management (°C).

    Parameters
    ----------
    temperature : float
        Initial temperature in Celsius.

    Notes
    -----
    * Values are clamped to [MIN_TEMPERATURE, MAX_TEMPERATURE].
    * Values must be real numbers (ints/floats); non-numerics raise TypeError.
    """

    def __init__(self, temperature: float = 37.0):
        self._temperature = self._validate_and_clamp(temperature)

    # --- properties ---------------------------------------------------------
    @property
    def temperature(self) -> float:
        """Current temperature (°C)."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature = self._validate_and_clamp(value)

    # --- API ----------------------------------------------------------------
    def set_temperature(self, value: float) -> None:
        """Set temperature (°C) with validation + clamping."""
        self._temperature = self._validate_and_clamp(value)

    def get_temperature(self) -> float:
        """Return current temperature (°C)."""
        return self._temperature

    # --- helpers ------------------------------------------------------------
    def _validate_and_clamp(self, value: float) -> float:
        """Validate numeric type and clamp to [MIN_T, MAX_T]."""
        if not isinstance(value, numbers.Real):
            raise TypeError(f"Temperature must be a real number, got {type(value).__name__}")
        v = float(value)
        if v < MIN_TEMPERATURE:
            v = MIN_TEMPERATURE
        elif v > MAX_TEMPERATURE:
            v = MAX_TEMPERATURE
        return v

    # --- serialization ------------------------------------------------------
    def to_json(self, full_model: bool = False) -> str:
        """
        Serialize to JSON.

        Parameters
        ----------
        full_model : bool
            If True, include a tiny wrapper object compatible with other exports.

        Returns
        -------
        str
        """
        payload: Dict[str, Any] = {"temperature": self._temperature}
        if full_model:
            payload = {"environment": {"temperature": self._temperature}}
        return json.dumps(payload, indent=2)

    def get_complete_model(self) -> Dict[str, Any]:
        """Return a dict useful for UI composition."""
        return {"environment": {"temperature": self._temperature}}

    # --- repr/str -----------------------------------------------------------
    def __repr__(self) -> str:
        return f"Temperature({self._temperature:.3g}°C)"

    def __str__(self) -> str:
        return f"{self._temperature:.2f}°C"
