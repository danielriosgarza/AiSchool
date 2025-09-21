# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
kinetic_model.stirring
~~~~~~~~~~~~~~~~~~~~~~~~~

Defines :class:`~kinetic_model.stirring.Stirring`, which manages
stirring/mixing effects in the kinetic model system.

A **Stirring** instance
- maintains a stirring rate from 0 to 1 (0 = poor mixing, 1 = perfect mixing)
- has a configurable base standard deviation for noise generation
- applies stirring effects to scalar or vector inputs
- returns samples with noise proportional to (1 - rate) * base_std
- ensures all samples are non-negative through truncation
- supports JSON serialization for model configuration

Example
-------
>>> from kinetic_model.stirring import Stirring
>>> import numpy as np
>>> stirring = Stirring(rate=0.5, base_std=0.1)
>>> stirring.apply_stirring(10.0)          # scalar OK
>>> stirring.apply_stirring(np.array([1.0, 2.0, 3.0]))  # vector OK

>>> # Apply to vector input
>>> input_vector = np.array([5.0, 10.0, 15.0])
>>> result_vector = stirring.apply_stirring(input_vector)
>>> print(result_vector)  # Samples around input values
>>> 
>>> # Perfect mixing (rate = 1)
>>> stirring.rate = 1.0
>>> result = stirring.apply_stirring(input_value)
>>> print(result)  # Exactly 10.0 (no noise)
>>> 
>>> # Poor mixing (rate = 0)
>>> stirring.rate = 0.0
>>> result = stirring.apply_stirring(input_value)
>>> print(result)  # Sample with std = 0.1
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
from __future__ import annotations

import json
import numpy as np
from typing import Union, List, Optional


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# Stirring rate bounds
MIN_RATE = 0.0
MAX_RATE = 1.0

# Default values
DEFAULT_RATE = 1.0  # Perfect mixing by default
DEFAULT_BASE_STD = 0.1  # Default base standard deviation


# --------------------------------------------------------------------------- #
# Public class
# --------------------------------------------------------------------------- #
class Stirring:
    """
    Represents stirring effects with configurable rate and noise.
    
    The class maintains a stirring rate and base standard deviation, then
    applies proportional noise to input values based on the rate.
    
    Parameters
    ----------
    rate : float
        Stirring rate from 0 to 1. 0 = poor mixing (maximum noise),
        1 = perfect mixing (no noise).
    base_std : float
        Base standard deviation for noise generation when rate = 0.
        
    Attributes
    ----------
    rate : float
        Current stirring rate (0.0 to 1.0).
    base_std : float
        Base standard deviation for noise generation.
    apply_stirring : function
        A function that applies stirring effects to input values.
        
    Notes
    -----
    * The effective standard deviation is (1 - rate) * base_std.
    * When rate = 1, no noise is applied (perfect mixing).
    * When rate = 0, maximum noise is applied (base_std).
    * All samples are truncated to ensure non-negative values.
    * The apply_stirring function is automatically created and assigned.
    """
    
    def __init__(self, rate: float = DEFAULT_RATE, base_std: float = DEFAULT_BASE_STD):
        """
        Initialize the Stirring object.
        
        Parameters
        ----------
        rate : float, optional
            Initial stirring rate (0.0 to 1.0). Defaults to 1.0 (perfect mixing).
        base_std : float, optional
            Base standard deviation for noise. Defaults to 0.1.
        """
        self._rate = self._validate_rate(rate)
        self._base_std = self._validate_base_std(base_std)
        
        # Create the stirring function
        self.apply_stirring = self._create_stirring_function()
    
    @property
    def rate(self) -> float:
        """
        Get the current stirring rate.
        
        Returns
        -------
        float
            Current stirring rate (0.0 to 1.0).
        """
        return self._rate
    
    @rate.setter
    def rate(self, value: float) -> None:
        """
        Set the stirring rate with validation.
        
        Parameters
        ----------
        value : float
            New stirring rate (0.0 to 1.0).
        """
        self._rate = self._validate_rate(value)
        # Recreate the stirring function with new rate
        self.apply_stirring = self._create_stirring_function()
    
    @property
    def base_std(self) -> float:
        """
        Get the base standard deviation.
        
        Returns
        -------
        float
            Base standard deviation for noise generation.
        """
        return self._base_std
    
    @base_std.setter
    def base_std(self, value: float) -> None:
        """
        Set the base standard deviation with validation.
        
        Parameters
        ----------
        value : float
            New base standard deviation.
        """
        self._base_std = self._validate_base_std(value)
        # Recreate the stirring function with new base_std
        self.apply_stirring = self._create_stirring_function()
    
    def set_rate(self, value: float) -> None:
        """
        Set the stirring rate with validation.
        
        Parameters
        ----------
        value : float
            New stirring rate (0.0 to 1.0).
        """
        self.rate = value
    
    def set_base_std(self, value: float) -> None:
        """
        Set the base standard deviation with validation.
        
        Parameters
        ----------
        value : float
            New base standard deviation.
        """
        self.base_std = value
    
    def get_rate(self) -> float:
        """
        Get the current stirring rate.
        
        Returns
        -------
        float
            Current stirring rate (0.0 to 1.0).
        """
        return self._rate
    
    def get_base_std(self) -> float:
        """
        Get the base standard deviation.
        
        Returns
        -------
        float
            Base standard deviation for noise generation.
        """
        return self._base_std
    
    def _validate_rate(self, value: float) -> float:
        """
        Validate and clamp stirring rate to valid range.
        
        Parameters
        ----------
        value : float
            Stirring rate to validate and clamp.
            
        Returns
        -------
        float
            Clamped stirring rate within valid range.
            
        Raises
        ------
        TypeError
            If value is not a number.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Stirring rate must be a number, got {type(value).__name__}")
        
        # Clamp to valid range
        clamped_value = max(MIN_RATE, min(MAX_RATE, float(value)))
        
        # Warn if value was clamped
        if clamped_value != value:
            print(f"Warning: Stirring rate {value} was clamped to {clamped_value}")
        
        return clamped_value
    
    def _validate_base_std(self, value: float) -> float:
        """
        Validate base standard deviation.
        
        Parameters
        ----------
        value : float
            Base standard deviation to validate.
            
        Returns
        -------
        float
            Validated base standard deviation.
            
        Raises
        ------
        TypeError
            If value is not a number.
        ValueError
            If value is negative.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Base standard deviation must be a number, got {type(value).__name__}")
        
        if value < 0:
            raise ValueError("Base standard deviation cannot be negative")
        
        return float(value)
    
    def _create_stirring_function(self):
        """
        Create the stirring function based on current rate and base_std.
        
        Returns
        -------
        function
            A function that applies stirring effects to input values.
        """
        rate = self._rate
        base_std = self._base_std
        
        def stirring_function(input_value: Union[float, np.ndarray, List[float]]) -> Union[float, np.ndarray]:
            """
            Apply stirring effects to input value(s).
            
            Parameters
            ----------
            input_value : float, np.ndarray, or List[float]
                Input value(s) to apply stirring to.
                
            Returns
            -------
            float or np.ndarray
                Stirred value(s) with applied noise.
            """
            # If perfect mixing (rate = 1), return input unchanged
            if rate == 1.0:
                return input_value
            
            # Calculate effective standard deviation
            effective_std = (1.0 - rate) * base_std
            
            # If no noise (effective_std = 0), return input unchanged
            if effective_std == 0.0:
                return input_value
            
            # Convert input to numpy array for processing
            if isinstance(input_value, (list, tuple)):
                input_array = np.array(input_value, dtype=float)
                return_scalar = False
            elif isinstance(input_value, np.ndarray):
                input_array = input_value.astype(float)
                return_scalar = False
            else:
                # Scalar input
                input_array = np.array([float(input_value)])
                return_scalar = True
            
            # Generate noise with normal distribution
            noise = np.random.normal(0, effective_std, input_array.shape)
            
            # Apply noise and ensure non-negative values
            stirred_values = np.maximum(0, input_array + noise)
            
            # Return scalar if input was scalar
            if return_scalar:
                return float(stirred_values[0])
            else:
                return stirred_values
        
        return stirring_function
    
    def to_json(self, full_model: bool = False) -> str:
        """
        Export stirring configuration to JSON.
        
        Parameters
        ----------
        full_model : bool, optional
            If True, includes minimal model structure. If False, only stirring config.
            Defaults to False.
            
        Returns
        -------
        str
            JSON string representation of the stirring configuration.
        """
        if full_model:
            # Return minimal model structure for integration
            model_data = {
                "type": "stirring",
                "rate": self._rate,
                "base_std": self._base_std
            }
        else:
            # Return just the stirring configuration
            model_data = {
                "rate": self._rate,
                "base_std": self._base_std
            }
        
        return json.dumps(model_data, indent=2)
    
    def get_complete_model(self) -> dict:
        """
        Get complete model data for visualization and integration.
        
        Returns
        -------
        dict
            Complete model data including type, rate, and base_std.
        """
        return {
            "type": "stirring",
            "rate": self._rate,
            "base_std": self._base_std
        }
    
    def __repr__(self) -> str:
        """String representation of the Stirring object."""
        return f"Stirring(rate={self._rate}, base_std={self._base_std})"
    
    def __str__(self) -> str:
        """String representation of the Stirring object."""
        return f"Stirring(rate={self._rate:.2f}, std={self._base_std:.3f})"
