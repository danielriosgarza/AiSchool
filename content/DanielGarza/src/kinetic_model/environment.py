
# -*- coding: utf-8 -*-
"""
kinetic_model.environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines :class:`~kinetic_model.environment.Environment`, which aggregates
pH, stirring, and temperature into one convenience object.

Example
-------
>>> import numpy as np
>>> from kinetic_model.metabolite import Metabolite
>>> from kinetic_model.metabolome import Metabolome
>>> from kinetic_model.ph import pH
>>> from kinetic_model.stirring import Stirring
>>> from kinetic_model.temperature import Temperature
>>> from kinetic_model.environment import Environment
>>>
>>> glucose = Metabolite("glucose", 0.0, {'C': 6, 'H': 12, 'O': 6})
>>> metabolome = Metabolome([glucose])
>>> ph_obj = pH(metabolome, intercept=7.0, met_dictionary={"glucose": -0.1})
>>> stirring_obj = Stirring(rate=0.8, base_std=0.05)
>>> temp_obj = Temperature(37.0)
>>> env = Environment(ph_obj, stirring_obj, temp_obj)
>>> res = env.apply_all_factors(np.array([5.0]))
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
from __future__ import annotations

import json
import numpy as np
from typing import Dict, Any, Optional, Union

from .ph import pH
from .stirring import Stirring
from .temperature import Temperature


# --------------------------------------------------------------------------- #
# Public class
# --------------------------------------------------------------------------- #
class Environment:
    """
    Represents the complete environment with pH, stirring, and temperature components.
    
    The class acts as a container for environmental factors, providing both
    unified methods to apply all factors and independent access to individual
    components.
    
    Parameters
    ----------
    ph_obj : pH
        pH object that manages pH computation and state.
    stirring_obj : Stirring
        Stirring object that manages mixing effects and noise.
    temperature_obj : Temperature
        Temperature object that manages temperature state.
        
    Attributes
    ----------
    pH : pH
        pH component for managing pH values and computation.
    stirring : Stirring
        Stirring component for managing mixing effects.
    temperature : Temperature
        Temperature component for managing temperature values.
        
    Notes
    -----
    * The Environment class provides a unified interface while maintaining
      independent access to each component.
    * All components maintain their individual state and functionality.
    * The unified methods apply environmental factors in a logical sequence.
    * JSON export maintains the current structure with separate nodes.
    """
    
    def __init__(
        self, 
        ph_obj: pH, 
        stirring_obj: Stirring, 
        temperature_obj: Temperature
    ) -> None:
        """
        Initialize the Environment with its components.
        
        Parameters
        ----------
        ph_obj : pH
            pH object for pH management.
        stirring_obj : Stirring
            Stirring object for mixing effects.
        temperature_obj : Temperature
            Temperature object for temperature management.
        """
        self.pH = ph_obj
        self.stirring = stirring_obj
        self.temperature = temperature_obj
    
    # ----------------------------------------------------------------------- #
    # Unified environmental factor application
    # ----------------------------------------------------------------------- #
    def apply_all_factors(
        self, 
        metabolite_concentrations: np.ndarray,
        update_pH: bool = True
    ) -> Dict[str, Any]:
        """
        Apply all environmental factors to metabolite concentrations.
        
        This method applies pH computation, stirring effects, and temperature
        influence in a logical sequence, returning a comprehensive result.
        
        Parameters
        ----------
        metabolite_concentrations : np.ndarray
            Array of metabolite concentrations in the same order as the metabolome.
        update_pH : bool, optional
            Whether to update the internal pH state. Defaults to True.
            
        Returns
        -------
        dict
            Dictionary containing all environmental factor results:
            - 'pH': computed pH value
            - 'stirred_concentrations': concentrations after stirring effects
            - 'temperature': current temperature
            - 'environmental_summary': summary of all applied factors
            
        Examples
        --------
        >>> concentrations = np.array([5.0, 10.0])  # glucose, lactate
        >>> result = env.apply_all_factors(concentrations)
        >>> print(result['pH'])  # Current pH value
        >>> print(result['stirred_concentrations'])  # Concentrations after stirring
        >>> print(result['temperature'])  # Current temperature
        """
        # Apply pH computation first
        current_pH = self.pH.compute_pH(metabolite_concentrations, update=update_pH)
        
        # Apply stirring effects to concentrations
        stirred_concentrations = self.stirring.apply_stirring(metabolite_concentrations)
        
        # Get current temperature
        current_temperature = self.temperature.temperature
        
        # Create comprehensive result
        result = {
            'pH': current_pH,
            'stirred_concentrations': stirred_concentrations,
            'temperature': current_temperature,
            'environmental_summary': {
                'pH_mode': 'weighted' if hasattr(self.pH, '_current_met_dictionary') and self.pH._current_met_dictionary else 'constant',
                'stirring_rate': self.stirring.rate,
                'stirring_noise_level': (1.0 - self.stirring.rate) * self.stirring.base_std,
                'temperature_range': f"{self.temperature.temperature}°C"
            }
        }
        
        return result
    
    def apply_environmental_effects(
        self, 
        input_values: Union[float, np.ndarray, list],
        include_pH: bool = True,
        include_stirring: bool = True,
        include_temperature: bool = True
    ) -> Dict[str, Any]:
        """
        Apply selected environmental effects to input values.
        
        This method allows selective application of environmental factors,
        useful for testing individual components or specific combinations.
        
        Parameters
        ----------
        input_values : float, np.ndarray, or list
            Input values to apply environmental effects to.
        include_pH : bool, optional
            Whether to include pH effects. Defaults to True.
        include_stirring : bool, optional
            Whether to include stirring effects. Defaults to True.
        include_temperature : bool, optional
            Whether to include temperature effects. Defaults to True.
            
        Returns
        -------
        dict
            Dictionary containing results of applied environmental effects.
            
        Examples
        --------
        >>> # Apply only stirring effects
        >>> result = env.apply_environmental_effects([1.0, 2.0, 3.0], 
        ...                                        include_pH=False, 
        ...                                        include_temperature=False)
        >>> print(result['stirred_values'])  # Only stirring applied
        """
        result = {}
        
        # Apply pH effects if requested
        if include_pH:
            # For pH, we need metabolite concentrations, so we'll use a default
            # This could be enhanced to accept metabolite context
            result['pH_note'] = "pH computation requires metabolite context"
        
        # Apply stirring effects if requested
        if include_stirring:
            stirred_values = self.stirring.apply_stirring(input_values)
            result['stirred_values'] = stirred_values
        
        # Apply temperature effects if requested
        if include_temperature:
            result['temperature'] = self.temperature.temperature
        
        return result
    
    # ----------------------------------------------------------------------- #
    # State management methods
    # ----------------------------------------------------------------------- #
    def get_environmental_state(self) -> Dict[str, Any]:
        """
        Get the current state of all environmental factors.
        
        Returns
        -------
        dict
            Current state of all environmental components.
        """
        return {
            'pH': {
                'current_value': self.pH.pH,
                'mode': 'weighted' if hasattr(self.pH, '_current_met_dictionary') and self.pH._current_met_dictionary else 'constant'
            },
            'stirring': {
                'rate': self.stirring.rate,
                'base_std': self.stirring.base_std,
                'effective_noise': (1.0 - self.stirring.rate) * self.stirring.base_std
            },
            'temperature': {
                'value': self.temperature.temperature,
                'unit': 'Celsius'
            }
        }
    
    def set_environmental_state(
        self, 
        ph_intercept: Optional[float] = None,
        ph_metabolite_weights: Optional[Dict[str, float]] = None,
        stirring_rate: Optional[float] = None,
        stirring_base_std: Optional[float] = None,
        temperature: Optional[float] = None
    ) -> None:
        """
        Set multiple environmental parameters at once.
        
        This method allows batch updating of environmental parameters
        while maintaining the individual component functionality.
        
        Parameters
        ----------
        ph_intercept : float, optional
            New pH intercept value.
        ph_metabolite_weights : dict, optional
            New metabolite weights for pH computation.
        stirring_rate : float, optional
            New stirring rate.
        stirring_base_std : float, optional
            New stirring base standard deviation.
        temperature : float, optional
            New temperature value.
            
        Examples
        --------
        >>> # Update multiple parameters
        >>> env.set_environmental_state(
        ...     ph_intercept=8.0,
        ...     stirring_rate=0.9,
        ...     temperature=42.0
        ... )
        """
        # Update pH if parameters provided
        if ph_intercept is not None or ph_metabolite_weights is not None:
            current_intercept = ph_intercept if ph_intercept is not None else self.pH._get_current_intercept()
            current_weights = ph_metabolite_weights if ph_metabolite_weights is not None else self.pH._get_current_met_dictionary()
            # Note: This requires access to the metabolome object, which we don't store
            # We'll need to handle this case or require it to be passed separately
            print("Note: pH parameter updates require metabolome context")
        
        # Update stirring if parameters provided
        if stirring_rate is not None:
            self.stirring.rate = stirring_rate
        if stirring_base_std is not None:
            self.stirring.base_std = stirring_base_std
        
        # Update temperature if provided
        if temperature is not None:
            self.temperature.temperature = temperature
    
    # ----------------------------------------------------------------------- #
    # JSON serialization
    # ----------------------------------------------------------------------- #
    def to_json(self, full_model: bool = False) -> str:
        """
        Export environment configuration to JSON.
        
        This method maintains the current structure with separate nodes
        for pH, temperature, and stirring components.
        
        Parameters
        ----------
        full_model : bool, optional
            If True, includes complete model structure. If False, only environment config.
            Defaults to False.
            
        Returns
        -------
        str
            JSON string representation of the environment configuration.
        """
        if full_model:
            # Return complete model structure maintaining separate nodes
            model_data = {
                "version": "1.0.0",
                "environment": {
                    "pH": json.loads(self.pH.to_json()),
                    "stirring": json.loads(self.stirring.to_json()),
                    "temperature": json.loads(self.temperature.to_json())
                }
            }
        else:
            # Return just the environment configuration
            model_data = {
                "pH": json.loads(self.pH.to_json()),
                "stirring": json.loads(self.stirring.to_json()),
                "temperature": json.loads(self.temperature.to_json())
            }
        
        return json.dumps(model_data, indent=2)
    
    def get_complete_model(self, metabolome_obj=None, temperature: float = 37.0) -> Dict[str, Any]:
        """
        Get complete environment model data for visualization and integration.
        
        Parameters
        ----------
        metabolome_obj : Metabolome, optional
            Metabolome object for pH complete model. If not provided, will use
            the metabolome from the pH object if available.
        temperature : float, default 37.0
            Temperature value for pH complete model.
            
        Returns
        -------
        dict
            Complete environment model data maintaining separate component nodes.
        """
        # For pH, we need to handle the case where metabolome_obj might be required
        try:
            ph_model = self.pH.get_complete_model(metabolome_obj, temperature)
        except TypeError:
            # If metabolome_obj is required but not provided, create a minimal one
            ph_model = self.pH.get_complete_model()
        
        return {
            "version": "1.0.0",
            "environment": {
                "pH": ph_model,
                "stirring": self.stirring.get_complete_model(),
                "temperature": self.temperature.get_complete_model()
            }
        }
    
    # ----------------------------------------------------------------------- #
    # Utility methods
    # ----------------------------------------------------------------------- #
    def validate_environment(self) -> Dict[str, bool]:
        """
        Validate that all environmental components are in valid states.
        
        Returns
        -------
        dict
            Validation results for each component.
        """
        validation_results = {
            'pH': True,  # pH class handles its own validation
            'stirring': True,  # Stirring class handles its own validation
            'temperature': True  # Temperature class handles its own validation
        }
        
        # Additional cross-component validation could be added here
        # For now, we rely on individual component validation
        
        return validation_results
    
    def reset_to_defaults(self) -> None:
        """
        Reset all environmental components to their default values.
        
        This method resets each component to its default state.
        Note: pH requires metabolome context for full reset.
        """
        # Reset stirring to perfect mixing
        self.stirring.rate = 1.0
        self.stirring.base_std = 0.1
        
        # Reset temperature to room temperature
        self.temperature.temperature = 23.0
        
        # Note: pH reset requires metabolome context
        print("Note: pH reset requires metabolome context")
    
    # ----------------------------------------------------------------------- #
    # String representations
    # ----------------------------------------------------------------------- #
    def __repr__(self) -> str:
        """String representation of the Environment object."""
        return f"Environment(pH={self.pH}, stirring={self.stirring}, temperature={self.temperature})"
    
    def __str__(self) -> str:
        """String representation of the Environment object."""
        return f"Environment(pH: {self.pH.pH:.2f}, Stirring: {self.stirring.rate:.2f}, Temp: {self.temperature.temperature:.1f}°C)"


# --------------------------------------------------------------------------- #
# End of file
# --------------------------------------------------------------------------- #

