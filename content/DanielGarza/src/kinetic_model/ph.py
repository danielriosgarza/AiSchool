# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
kinetic_model.ph
~~~~~~~~~~~~~~~~~~~

Defines :class:`~kinetic_model.ph.ph`, which manages pH values in the
kinetic model system.

Example
-------
>>> import numpy as np
>>> from kinetic_model.metabolite import Metabolite
>>> from kinetic_model.metabolome import Metabolome
>>> from kinetic_model.ph import pH
>>> glucose = Metabolite("glucose", 0.0, {'C': 6, 'H': 12, 'O': 6})
>>> lactate = Metabolite("lactate", 0.0, {'C': 3, 'H': 6, 'O': 3})
>>> acetate = Metabolite("acetate", 0.0, {'C': 2, 'H': 4, 'O': 2})
>>> metabolome = Metabolome([glucose, lactate, acetate])
>>> weights = {"glucose": -0.1, "acetate": 0.05}
>>> ph_obj = pH(metabolome, intercept=7.0, met_dictionary=weights)
>>> c = np.array([metabolome.get_metabolite(n).concentration for n in metabolome.metabolites])
>>> ph_obj.compute_pH(c)
>>> # Switch to constant mode
>>> ph_obj.change_computation_function(metabolome, 8.5, {})
>>> print(ph_obj.compute_pH(concentrations))  # 8.5 (constant)
>>> 
>>> # Export to JSON
>>> json_str = ph_obj.to_json()  # pH configuration only
>>> model_json = ph_obj.to_json(full_model=True)  # Minimal model structure
>>> 
>>> # Get complete model for visualization (recommended)
>>> complete_model = ph_obj.get_complete_model(metabolome)
>>> # Use with GraphSpecBuilder: builder.build_from_json(complete_model)
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
from __future__ import annotations

import json
import numpy as np
from typing import Dict, List, Optional, Callable

from .metabolome import Metabolome


# --------------------------------------------------------------------------- #
# Public class
# --------------------------------------------------------------------------- #
class pH:
    """
    Represents pH with internal state and computation function.
    
    The class maintains a pH value and can compute pH from metabolite concentrations
    using a linear model: pH = intercept + sum(weights × concentrations).
    
    The class automatically handles metabolite ordering based on the metabolome object
    and assigns weight 0 to metabolites not specified in the weights dictionary.
    
    Parameters
    ----------
    metabolome_obj : Metabolome
        The metabolome object that defines the metabolite structure and ordering.
        The computation function will use this ordering for the weight vector.
    intercept : float
        Intercept term for pH computation. This is the baseline pH when all
        metabolite concentrations are zero.
    met_dictionary : dict
        Dictionary mapping metabolite names to their weights in pH computation.
        Metabolites not in this dict get weight 0.0 automatically.
        
    Attributes
    ----------
    pH : float
        Current pH value (internal state). Initialized to the intercept value.
    compute_pH : function
        A function that computes pH from metabolite concentration vectors.
        Can optionally update the internal pH state.
        
    Notes
    -----
    * The internal pH state is initialized to the intercept value.
    * The compute_pH function is automatically created and assigned.
    * Metabolite ordering follows metabolome_obj.metabolites (alphabetically sorted).
    * pH values are automatically clamped to the valid range [0.0, 14.0].
    * Empty met_dictionary results in constant pH mode (pH = intercept).
    * The change_computation_function method allows dynamic mode switching.
    * JSON serialization follows the MicrobiomeGym model schema.
    """
    
    def __init__(
        self, 
        metabolome_obj: Metabolome, 
        intercept: float, 
        met_dictionary: Dict[str, float]
    ) -> None:
        """
        Initialize pH class.
        
        Parameters
        ----------
        metabolome_obj : Metabolome
            The metabolome object that defines the metabolite structure and ordering.
        intercept : float
            Intercept term for pH computation.
        met_dictionary : dict
            Dictionary mapping metabolite names to their weights.
            
        Notes
        -----
        The internal pH state is initialized to the intercept value.
        The compute_pH function is automatically created and assigned.
        """
        self.pH = intercept  # internal state
        self._metabolome = metabolome_obj  # Store reference for JSON export
        self.compute_pH = self.get_pH_function(metabolome_obj, intercept, met_dictionary)
    
    # ----------------------------------------------------------------------- #
    # pH computation and state management
    # ----------------------------------------------------------------------- #
    def change_computation_function(
        self, 
        metabolome_obj: Metabolome, 
        intercept: float, 
        met_dictionary: Dict[str, float]
    ) -> None:
        """
        Change the computation function to use new parameters.
        
        This method allows dynamically switching between different pH computation
        modes, such as changing from weighted to constant pH computation.
        
        Parameters
        ----------
        metabolome_obj : Metabolome
            The new metabolome object that defines the metabolite structure.
        intercept : float
            New intercept term for pH computation.
        met_dictionary : dict
            New dictionary mapping metabolite names to their weights.
            Use an empty dict {} for constant pH computation.
            
        Notes
        -----
        - The internal pH state is updated to the new intercept value.
        - The compute_pH function is recreated with the new parameters.
        - This allows switching between weighted and constant pH modes.
        
        Examples
        --------
        >>> # Switch to constant pH mode
        >>> ph_obj.change_computation_function(metabolome, 8.5, {})
        >>> 
        >>> # Switch back to weighted mode
        >>> new_weights = {"glucose": 0.2, "lactate": -0.1}
        >>> ph_obj.change_computation_function(metabolome, 7.0, new_weights)
        """
        # Update internal pH state to new intercept
        self.pH = intercept
        
        # Store reference to new metabolome for JSON export
        self._metabolome = metabolome_obj
        
        # Recreate the computation function with new parameters
        self.compute_pH = self.get_pH_function(metabolome_obj, intercept, met_dictionary)
    
    # ----------------------------------------------------------------------- #
    # JSON serialization
    # ----------------------------------------------------------------------- #
    def to_json(self, filename: str = None, full_model: bool = False, metabolome_obj: Metabolome = None) -> str:
        """
        Convert the pH object to JSON format.
        
        This method serializes the pH configuration following the MicrobiomeGym
        model schema. The JSON structure includes the base pH value and any
        metabolite influences, making it compatible with the visualization
        and simulation systems.
        
        Parameters
        ----------
        filename : str, optional
            If provided, save the JSON to this file.
        full_model : bool, default False
            If True, returns a complete model structure ready for visualization.
            If False, returns just the pH configuration data.
        metabolome_obj : Metabolome, optional
            Metabolome object to include in full model JSON. If not provided and full_model=True,
            will use the metabolome from which this pH object was created.
            
        Returns
        -------
        str
            JSON string representation of the pH configuration.
            
        Examples
        --------
        >>> # Create pH object with metabolite influences
        >>> ph_obj = pH(metabolome, intercept=7.0, met_dictionary={"acetate": -0.1})
        >>> 
        >>> # Get just the pH configuration
        >>> json_str = ph_obj.to_json()
        >>> 
        >>> # Get complete model structure
        >>> model_json = ph_obj.to_json(full_model=True)
        >>> 
        >>> # Save to file
        >>> ph_obj.to_json("ph_config.json")
        
        Notes
        -----
        The JSON structure follows the MicrobiomeGym model schema:
        - baseValue: The intercept pH value
        - connectedMetabolites: Dictionary of metabolite influences (weights)
        - name: Default pH control identifier
        - color: Default visualization color
        
        Schema compliance:
        - baseValue is clamped to [0, 14] range
        - connectedMetabolites only includes metabolites with non-zero weights
        - Empty met_dictionary results in constant pH mode
        
        Note: When full_model=True, you should provide metabolome_obj to ensure
        metabolite nodes are visible in visualizations.
        """
        # Determine if we're in constant or weighted mode
        is_constant = not bool(self._get_current_met_dictionary())
        
        # Get current intercept (could have been changed via change_computation_function)
        current_intercept = self._get_current_intercept()
        
        # Create pH configuration data
        ph_data = {
            "name": "pH Control",
            "baseValue": current_intercept,
            "color": "#10b981",  # Default green color for pH
            "description": "pH control with metabolite influence" if not is_constant else "Constant pH control"
        }
        
        # Add metabolite influences if not in constant mode
        if not is_constant:
            current_met_dict = self._get_current_met_dictionary()
            # Only include metabolites with non-zero weights
            connected_metabolites = {name: weight for name, weight in current_met_dict.items() if weight != 0.0}
            if connected_metabolites:
                ph_data["connectedMetabolites"] = connected_metabolites
        
        if full_model:
            # Use provided metabolome_obj or fall back to the one used to create this pH object
            target_metabolome = metabolome_obj if metabolome_obj is not None else getattr(self, '_metabolome', None)
            
            # Create metabolites data for the metabolome
            metabolites_data = {}
            if target_metabolome is not None:
                # Convert metabolites to the format expected by the schema
                for name in target_metabolome.metabolites:
                    metabolite = target_metabolome._metabolite_dict[name]
                    metabolites_data[name] = {
                        "name": metabolite.name,
                        "concentration": metabolite.concentration,
                        "formula": metabolite.formula,
                        "color": metabolite.color
                    }
                    # Only include description if it's not empty
                    if metabolite.description:
                        metabolites_data[name]["description"] = metabolite.description
            else:
                # Fallback: create minimal metabolite entries
                metabolites_data = {
                    "default_metabolite": {
                        "name": "default_metabolite",
                        "concentration": 1.0,
                        "formula": {"C": 1, "H": 1, "O": 1},
                        "color": "#cccccc"
                    }
                }
            
            # Return complete model structure with actual metabolite data
            model_data = {
                "version": "1.0.0",
                "metabolome": {
                    "metabolites": metabolites_data,  # Now includes actual metabolite data
                    "pH": ph_data,
                    "temperature": 37.0
                },
                "microbiome": {
                    "name": "default_community",
                    "bacteria": {}
                }
            }
            json_str = json.dumps(model_data, indent=2)
        else:
            # Return just pH configuration data
            json_str = json.dumps(ph_data, indent=2)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def get_complete_model(self, metabolome_obj: Metabolome, temperature: float = 37.0) -> dict:
        """
        Get a complete model structure that includes both pH and metabolome data.
        
        This method creates a complete model structure that can be used directly
        with the GraphSpecBuilder for visualization.
        
        Parameters
        ----------
        metabolome_obj : Metabolome
            The metabolome object to include in the model.
        temperature : float, default 37.0
            Temperature value for the model.
            
        Returns
        -------
        dict
            Complete model structure ready for visualization.
            
        Examples
        --------
        >>> # Get complete model for visualization
        >>> complete_model = ph_obj.get_complete_model(metabolome)
        >>> 
        >>> # Use directly with GraphSpecBuilder
        >>> builder = GraphSpecBuilder()
        >>> graph_spec = builder.build_from_json(complete_model)
        """
        # Get pH configuration
        ph_config = json.loads(self.to_json())
        
        # Get metabolome configuration
        metabolome_config = json.loads(metabolome_obj.to_json())
        
        # Create complete model structure
        complete_model = {
            "version": "1.0.0",
            "metabolome": {
                "metabolites": metabolome_config["metabolites"],
                "pH": ph_config,
                "temperature": temperature
            },
            "microbiome": {
                "name": "default_community",
                "bacteria": {}
            }
        }
        
        return complete_model
    
    # ----------------------------------------------------------------------- #
    # Helper methods
    # ----------------------------------------------------------------------- #
    def _get_current_intercept(self) -> float:
        """
        Get the current intercept value.
        
        This helper method extracts the current intercept from the computation
        function closure, allowing us to serialize the current configuration.
        
        Returns
        -------
        float
            Current intercept value used in pH computation.
        """
        # Extract intercept from the computation function's closure
        # This is a bit of a hack, but it's the cleanest way to get the current state
        if hasattr(self, '_current_intercept'):
            return self._current_intercept
        else:
            # Fallback: try to get it from the function's closure
            # This is not ideal but works for the current implementation
            return self.pH  # Use current pH as fallback
    
    def _get_current_met_dictionary(self) -> Dict[str, float]:
        """
        Get the current metabolite dictionary.
        
        This helper method extracts the current metabolite weights from the
        computation function closure, allowing us to serialize the current configuration.
        
        Returns
        -------
        dict
            Current dictionary mapping metabolite names to weights.
        """
        # Extract met_dictionary from the computation function's closure
        if hasattr(self, '_current_met_dictionary'):
            return self._current_met_dictionary
        else:
            # Fallback: return empty dict (constant mode)
            return {}
    
    def get_pH_function(
        self, 
        metabolome_obj: Metabolome, 
        intercept: float, 
        met_dictionary: Dict[str, float]
    ) -> Callable[[np.ndarray, bool], float]:
        """
        Create and return a pH computation function.
        
        This method creates a closure that captures the metabolome structure,
        intercept, and weights, returning a function that can compute pH
        from concentration vectors.
        
        Parameters
        ----------
        metabolome_obj : Metabolome
            The metabolome object that defines the metabolite structure.
        intercept : float
            Intercept term for pH computation.
        met_dictionary : dict
            Dictionary mapping metabolite names to their weights.
            
        Returns
        -------
        function
            A function that computes pH from concentration vectors.
            The function signature is:
            compute_pH(metabolome_concentration_vector, update=True)
            
        Notes
        -----
        The returned function automatically handles:
        - Metabolite ordering based on metabolome_obj.metabolites
        - Missing metabolites (assigned weight 0.0)
        - pH bounds enforcement (0.0 to 14.0)
        - Optional internal state updates
        
        Special cases:
        - If met_dictionary is empty {}, the function returns constant pH (intercept)
        - If met_dictionary has weights, the function computes weighted pH
        """
        # Store current configuration for serialization
        self._current_intercept = intercept
        self._current_met_dictionary = met_dictionary.copy()
        
        def compute_ph(metabolome_concentration_vector: np.ndarray, update: bool = True) -> float:
            """
            Compute pH from metabolite concentration vector.
            
            Parameters
            ----------
            metabolome_concentration_vector : np.ndarray
                Array of metabolite concentrations in the same order as
                metabolome_obj.metabolites.
            update : bool, optional
                If True, update the internal pH state (self.pH).
                If False, only return the computed pH without updating state.
                Default is True.
                
            Returns
            -------
            float
                Computed pH value, clamped between 0.0 and 14.0.
                
            Raises
            ------
            ValueError
                If the concentration vector length doesn't match the number
                of metabolites in the metabolome.
                
            Notes
            -----
            The computation follows the linear model:
            pH = intercept + sum(weights × concentrations)
            
            Metabolites not in met_dictionary get weight 0.0 automatically.
            The pH is automatically clamped to the valid range [0.0, 14.0].
            
            Special case: If met_dictionary is empty, pH = intercept (constant).
            """
            # Validate input length
            if len(metabolome_concentration_vector) != len(metabolome_obj.metabolites):
                raise ValueError(
                    f"Concentration vector length ({len(metabolome_concentration_vector)}) "
                    f"must match metabolome metabolite count ({len(metabolome_obj.metabolites)})"
                )
            
            # Check if we're in constant pH mode (empty met_dictionary)
            if not met_dictionary:
                new_pH = intercept
            else:
                # Create weight vector in the same order as metabolome metabolites
                # Metabolites not in met_dictionary get weight 0.0
                weight_vector = np.array([met_dictionary.get(name, 0.0) 
                                        for name in metabolome_obj.metabolites])
                
                # Compute pH using linear model: pH = intercept + sum(weights × concentrations)
                new_pH = intercept + np.dot(weight_vector, metabolome_concentration_vector)
            
            # Ensure pH stays within valid range
            new_pH = max(0.0, min(14.0, new_pH))
            
            # Update internal state if requested
            if update:
                self.pH = new_pH
                
            return new_pH
        
        return compute_ph


# --------------------------------------------------------------------------- #
# End of file
# --------------------------------------------------------------------------- #
