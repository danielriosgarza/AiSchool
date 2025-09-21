# -*- coding: utf-8 -*-
"""
kinetic_model.feeding_term
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines :class:`~kinetic_model.feeding_term.FeedingTerm`, which represents
microbial phenotypes and their metabolic strategies.

Example
-------
>>> from kinetic_model.metabolite import Metabolite
>>> from kinetic_model.metabolome import Metabolome
>>> from kinetic_model.feeding_term import FeedingTerm


>>> glucose = Metabolite("glucose", 10.0, {'C': 6, 'H': 12, 'O': 6})
>>> lactate = Metabolite("lactate", 5.0, {'C': 3, 'H': 6, 'O': 3})
>>> metabolome = Metabolome(metabolites=[glucose, lactate])
>>> 
>>> # Create feeding term: consumes glucose, produces lactate
>>> feeding_term = FeedingTerm(
...     id="glucose_to_lactate",
...     metDict={"glucose": [1.0, 2.0], "lactate": [-0.5, 0.0]},
...     metabolome=metabolome
... )
>>> 
>>> # Get current concentrations
>>> concentrations = metabolome.get_concentration()
>>> 
>>> # Calculate growth and metabolism rates
>>> growth_rate = feeding_term.intrinsicGrowth(concentrations)
>>> metabolism_rates = feeding_term.intrinsicMetabolism(concentrations)
>>> print(f"Growth rate: {growth_rate:.3f}")
>>> print(f"Metabolism rates: {metabolism_rates}")

Advanced usage with larger metabolome and co-consumption:
>>> # Create a larger metabolome with 25 metabolites
>>> metabolites = []
>>> for i in range(25):
...     name = f"metabolite_{i:02d}"
...     conc = 1.0 + i * 0.1  # Different concentrations
...     formula = {'C': i % 5 + 1, 'H': (i % 3 + 1) * 2, 'O': i % 4 + 1}
...     metabolites.append(Metabolite(name, conc, formula))
>>> 
>>> large_metabolome = Metabolome(metabolites=metabolites)
>>> print(f"Metabolome size: {len(large_metabolome.metabolites)}")
>>> 
>>> # Create feeding term that only uses 3 metabolites for consumption and 4 for production
>>> # This demonstrates the ability to work with a subset of a larger metabolome
>>> feeding_term = FeedingTerm(
...     id="complex_metabolism",
...     metDict={
...         "metabolite_00": [1.0, 0.5],      # Consume metabolite_00
...         "metabolite_05": [0.8, 1.2],      # Consume metabolite_05  
...         "metabolite_12": [0.6, 0.8],      # Consume metabolite_12
...         "metabolite_03": [-0.4, 0.0],     # Produce metabolite_03
...         "metabolite_08": [-0.3, 0.0],     # Produce metabolite_08
...         "metabolite_15": [-0.2, 0.0],     # Produce metabolite_15
...         "metabolite_22": [-0.1, 0.0]      # Produce metabolite_22
...     },
...     metabolome=large_metabolome
... )
>>> 
>>> # Get concentrations from the large metabolome
>>> concentrations = large_metabolome.get_concentration()
>>> print(f"Concentration vector length: {len(concentrations)}")
>>> 
>>> # Calculate growth rate (co-consumption of 3 substrates with AND relationship)
>>> growth_rate = feeding_term.intrinsicGrowth(concentrations)
>>> print(f"Growth rate: {growth_rate:.6f}")
>>> 
>>> # Calculate metabolism rates for ALL 25 metabolites
>>> metabolism_rates = feeding_term.intrinsicMetabolism(concentrations)
>>> print(f"Metabolism rates length: {len(metabolism_rates)}")
>>> print(f"Non-zero metabolism rates: {np.count_nonzero(metabolism_rates)}")
>>> 
>>> # Verify that only the 7 metabolites in our feeding term have non-zero rates
>>> # (3 consumed + 4 produced = 7 total)
>>> assert np.count_nonzero(metabolism_rates) == 7
>>> 
>>> # Verify that consumed metabolites have negative rates (consumption)
>>> glucose_idx = large_metabolome.metabolites.index("metabolite_00")
>>> assert metabolism_rates[glucose_idx] < 0  # Consumed
>>> 
>>> # Verify that produced metabolites have positive rates (production)  
>>> lactate_idx = large_metabolome.metabolites.index("metabolite_03")
>>> assert metabolism_rates[lactate_idx] > 0  # Produced
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Any


class FeedingTerm:
    """
    Represents a microbial feeding phenotype defining metabolite consumption and production.
    
    A FeedingTerm describes how a microorganism interacts with metabolites in its environment.
    It has two core components:
    
    1. **Intrinsic Growth**: Calculates growth rate based on consumed metabolites using 
       Monod kinetics with AND relationships (all consumed metabolites must be available).
    2. **Intrinsic Metabolism**: Calculates metabolism rates proportional to growth rate 
       and yield coefficients.
    
    The FeedingTerm is created from a Metabolome object to learn the structure and ordering,
    but calculates intrinsic growth and metabolism from a simple vector of concentrations
    that maintains the same dimensions as the metabolome's metabolites.
    
    Parameters
    ----------
    id : str
        Unique identifier for the feeding term.
    metDict : Dict[str, Tuple[float, float]]
        Dictionary mapping metabolite names to (yield, monodK) tuples.
        - Positive yields indicate consumption
        - Negative yields indicate production
        - monodK is the Monod constant for growth kinetics
    metabolome : Metabolome
        Metabolome object that defines the metabolite structure and ordering.
        
    Attributes
    ----------
    id : str
        Feeding term identifier.
    metabolite_names : List[str]
        List of metabolite names in metabolome order.
    yields : np.ndarray
        Array of yield coefficients (positive = consumption, negative = production).
    monodKs : np.ndarray
        Array of Monod constants for growth kinetics.
    intrinsicGrowth : Callable
        Function that calculates growth rate from concentration vector.
    intrinsicMetabolism : Callable
        Function that calculates metabolism rates from concentration vector.
        
    Examples
    --------
    Basic usage with glucose consumption and lactate production:
    
    >>> from microbiome_gym.kinetic_model import Metabolite, Metabolome, FeedingTerm
    >>> 
    >>> # Create metabolites
    >>> glucose = Metabolite("glucose", 10.0, {'C': 6, 'H': 12, 'O': 6})
    >>> lactate = Metabolite("lactate", 0.0, {'C': 3, 'H': 6, 'O': 3})
    >>> 
    >>> # Create metabolome
    >>> metabolome = Metabolome(metabolites=[glucose, lactate])
    >>> 
    >>> # Create feeding term: consume glucose, produce lactate
    >>> feeding_term = FeedingTerm(
    ...     id="glucose_fermentation",
    ...     metDict={
    ...         "glucose": (1.0, 0.5),    # Consume glucose with monodK=0.5
    ...         "lactate": (-0.8, 0.0)    # Produce lactate (monodK ignored)
    ...     },
    ...     metabolome=metabolome
    ... )
    >>> 
    >>> # Get current concentrations
    >>> concentrations = metabolome.get_concentration()
    >>> 
    >>> # Calculate growth rate (glucose availability)
    >>> growth_rate = feeding_term.intrinsicGrowth(concentrations)
    >>> print(f"Growth rate: {growth_rate:.3f}")
    >>> 
    >>> # Calculate metabolism rates
    >>> metabolism_rates = feeding_term.intrinsicMetabolism(concentrations)
    >>> print(f"Glucose consumption: {metabolism_rates[0]:.3f}")
    >>> print(f"Lactate production: {metabolism_rates[1]:.3f}")
    
    Co-consumption example (multiple substrates required):
    
    >>> # Create metabolites for aerobic respiration
    >>> glucose = Metabolite("glucose", 10.0, {'C': 6, 'H': 12, 'O': 6})
    >>> oxygen = Metabolite("oxygen", 5.0, {'O': 2})
    >>> co2 = Metabolite("co2", 0.0, {'C': 1, 'O': 2})
    >>> 
    >>> metabolome = Metabolome(metabolites=[glucose, oxygen, co2])
    >>> 
    >>> # Both glucose AND oxygen required for growth (AND relationship)
    >>> feeding_term = FeedingTerm(
    ...     id="aerobic_respiration",
    ...     metDict={
    ...         "glucose": (1.0, 1.0),    # Consume glucose
    ...         "oxygen": (0.3, 0.5),     # Consume oxygen
    ...         "co2": (-1.2, 0.0)        # Produce CO2
    ...     },
    ...     metabolome=metabolome
    ... )
    >>> 
    >>> # Growth rate is limited by BOTH glucose AND oxygen availability
    >>> concentrations = metabolome.get_concentration()
    >>> growth_rate = feeding_term.intrinsicGrowth(concentrations)
    >>> print(f"Growth rate: {growth_rate:.3f}")
    
    Large metabolome with subset usage:
    
    >>> # Create a large metabolome with 25 metabolites
    >>> metabolites = []
    >>> for i in range(25):
    ...     name = f"metabolite_{i:02d}"
    ...     conc = 1.0 + i * 0.1
    ...     formula = {'C': i % 5 + 1, 'H': (i % 3 + 1) * 2, 'O': i % 4 + 1}
    ...     metabolites.append(Metabolite(name, conc, formula))
    >>> 
    >>> large_metabolome = Metabolome(metabolites=metabolites)
    >>> 
    >>> # Feeding term only uses 3 metabolites for consumption and 4 for production
    >>> feeding_term = FeedingTerm(
    ...     id="complex_metabolism",
    ...     metDict={
    ...         "metabolite_00": [1.0, 0.5],      # Consume metabolite_00
    ...         "metabolite_05": [0.8, 1.2],      # Consume metabolite_05  
    ...         "metabolite_12": [0.6, 0.8],      # Consume metabolite_12
    ...         "metabolite_03": [-0.4, 0.0],     # Produce metabolite_03
    ...         "metabolite_08": [-0.3, 0.0],     # Produce metabolite_08
    ...         "metabolite_15": [-0.2, 0.0],     # Produce metabolite_15
    ...         "metabolite_22": [-0.1, 0.0]      # Produce metabolite_22
    ...     },
    ...     metabolome=large_metabolome
    ... )
    >>> 
    >>> # Calculate from full 25-element concentration vector
    >>> concentrations = large_metabolome.get_concentration()
    >>> growth_rate = feeding_term.intrinsicGrowth(concentrations)
    >>> metabolism_rates = feeding_term.intrinsicMetabolism(concentrations)
    >>> 
    >>> # Only 7 metabolites have non-zero metabolism rates
    >>> assert np.count_nonzero(metabolism_rates) == 7
    
    JSON serialization:
    
    >>> # Export to JSON format
    >>> json_str = feeding_term.to_json()
    >>> print("Basic JSON:", json_str)
    >>> 
    >>> # Export complete model structure
    >>> model_json = feeding_term.to_json(full_model=True)
    >>> print("Full model:", model_json)
    >>> 
    >>> # Save to file
    >>> feeding_term.to_json("feeding_term.json")
    
    Notes
    -----
    - Growth rate calculation uses Monod kinetics: concentration / (concentration + monodK)
    - Multiple consumed metabolites use AND logic (multiplied together)
    - Produced metabolites (negative yields) don't affect growth rate
    - Monod K values for produced metabolites are automatically set to 0
    - Concentration vectors must match the metabolome size exactly
    - The class automatically handles metabolite ordering based on the metabolome
    """
    
    def __init__(self, id: str, metDict: Dict[str, Tuple[float, float]], 
                 metabolome):
        """
        Initialize a FeedingTerm.
        
        Parameters
        ----------
        id : str
            Name/identifier of the functional term
        metDict : Dict[str, Tuple[float, float]]
            Dictionary mapping metabolite IDs to (yield, monodK) tuples.
            All metabolites in this dict will be considered with AND relationships
            (thus, multiplied together).
        metabolome
            Metabolome object to learn the metabolite structure and ordering.
            The feeding term will be compatible with concentration vectors that
            match this metabolome's metabolite order.
            
        Raises
        ------
        ValueError
            If id is empty, metDict is empty, or monodK is not positive for consumed metabolites
        TypeError
            If id is not a string or metDict is not a dictionary
        """
        # Validate inputs
        if not isinstance(id, str):
            raise TypeError("id must be a string")
        if not id.strip():
            raise ValueError("id cannot be empty")
        if not isinstance(metDict, dict):
            raise TypeError("metDict must be a dictionary")
        if not metDict:
            raise ValueError("metDict cannot be empty")
            
        # Validate metDict structure
        for met_id, values in metDict.items():
            if not isinstance(met_id, str):
                raise TypeError(f"metabolite ID must be string, got {type(met_id)}")
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise TypeError(f"metabolite values must be (yield, monodK) tuple or list, got {values}")
            if not isinstance(values[0], (int, float)) or not isinstance(values[1], (int, float)):
                raise TypeError(f"yield and monodK must be numbers, got {values}")
            
            yield_val, monodK = values
            
            # For produced metabolites (negative yields), Monod K should be 0
            if yield_val < 0:
                if monodK != 0:
                    import warnings
                    warnings.warn(
                        f"Warning: Metabolite '{met_id}' has negative yield ({yield_val}) indicating production. "
                        f"Monod K ({monodK}) will be ignored and set to 0 for produced metabolites.",
                        UserWarning
                    )
                    # Update the metDict to set Monod K to 0 for produced metabolites
                    metDict[met_id] = (yield_val, 0.0)
            else:
                # For consumed metabolites (positive yields), Monod K must be positive
                if monodK <= 0:
                    raise ValueError(f"monodK must be positive for consumed metabolites, got {monodK} for '{met_id}'")
        
        self.id = id
        
        # Store metabolite names in metabolome order
        self.metabolite_names = metabolome.metabolites.copy()
        
        # Store reference to metabolome for JSON export
        self._metabolome = metabolome
        
        # Initialize yields and monodKs arrays to match metabolome size
        self.yields = np.zeros(len(metabolome.metabolites))
        self.monodKs = np.zeros(len(metabolome.metabolites))
        
        # Fill in the actual values from metDict
        for met_id, (yield_val, monodK) in metDict.items():
            if met_id in self.metabolite_names:
                idx = self.metabolite_names.index(met_id)
                self.yields[idx] = yield_val
                self.monodKs[idx] = monodK
            else:
                raise ValueError(f"Metabolite '{met_id}' from metDict not found in metabolome")
        
        # Create the calculation functions
        self.intrinsicGrowth = self.__getIntrinsicGrowth()
        self.intrinsicMetabolism = self.__getIntrinsicMetabolism()
    
    # ----------------------------------------------------------------------- #
    # Growth and metabolism calculations
    # ----------------------------------------------------------------------- #
    
    def __getIntrinsicGrowth(self) -> Callable[[np.ndarray], float]:
        """
        Create a function that calculates intrinsic growth rate from concentration vector.
        
        Returns
        -------
        Callable[[np.ndarray], float]
            Function that takes a concentration vector and returns growth rate
        """
        
        def gr(concentrations):
            """
            Calculate growth rate from metabolite concentrations.
            
            Parameters
            ----------
            concentrations : np.ndarray
                Array of metabolite concentrations in the same order as metabolome.metabolites
                
            Returns
            -------
            float
                Growth rate (0.0 to 1.0)
                
            Raises
            ------
            ValueError
                If concentration vector length doesn't match metabolome size
            """
            if len(concentrations) != len(self.metabolite_names):
                raise ValueError(f"Concentration vector length ({len(concentrations)}) must match metabolome size ({len(self.metabolite_names)})")
            
            g = 1.0
            
            for i, yield_val in enumerate(self.yields):
                if yield_val > 0:  # Only consumed metabolites contribute to growth
                    concentration = concentrations[i]
                    monodK = self.monodKs[i]
                    
                    # Calculate Monod term for this substrate
                    denom = max(concentration + monodK, 1e-4)  # Avoid division by zero
                    frac = concentration / denom
                    
                    # Ensure fraction is non-negative and clamp to [0, 1]
                    frac = max(0.0, min(1.0, frac))
                    
                    # Apply AND logic: multiply by this substrate's contribution
                    g *= frac
                    
                    # If any substrate is missing, growth should be zero
                    if concentration == 0:
                        g = 0.0
                        break
            
            return g

        return gr
    
    def __getIntrinsicMetabolism(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create a function that calculates intrinsic metabolism rates from concentration vector.
        
        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Function that takes a concentration vector and returns metabolism rates
            with length equal to the number of metabolites in the metabolome
        """
        
        def metab(concentrations):
            """
            Calculate metabolism rates from metabolite concentrations.
            
            Parameters
            ----------
            concentrations : np.ndarray
                Array of metabolite concentrations in the same order as metabolome.metabolites
                
            Returns
            -------
            np.ndarray
                Array of metabolism rates (negative for consumption, positive for production)
                
            Raises
            ------
            ValueError
                If concentration vector length doesn't match metabolome size
            """
            if len(concentrations) != len(self.metabolite_names):
                raise ValueError(f"Concentration vector length ({len(concentrations)}) must match metabolome size ({len(self.metabolite_names)})")
            
            omega = self.intrinsicGrowth(concentrations)
            
            # Metabolism rate = -growth_rate * yield
            # Positive yields (consumed) → negative rates (consumption)
            # Negative yields (produced) → positive rates (production)
            return -omega * self.yields
        
        return metab
    
    # ----------------------------------------------------------------------- #
    # String representations
    # ----------------------------------------------------------------------- #
    
    def __repr__(self) -> str:
        """Return string representation of FeedingTerm."""
        return f"FeedingTerm(id='{self.id}', metabolites={self.metabolite_names}, yields={self.yields}, monodKs={self.monodKs})"
    
    def __str__(self) -> str:
        """Return human-readable string representation of FeedingTerm."""
        return f"FeedingTerm('{self.id}') with {len(self.metabolite_names)} metabolites"
    
    def get_data(self) -> Dict[str, Any]:
        """
        Get feeding term data as a dictionary (not JSON string).
        
        Returns
        -------
        dict
            Dictionary representation of the feeding term data.
        """
        feeding_term_data = {
            "id": self.id,
            "metDict": {},
            "metabolome_metabolites": self.metabolite_names.copy()
        }
        
        # Build metDict with only non-zero yields (active metabolites)
        for i, (met_name, yield_val) in enumerate(zip(self.metabolite_names, self.yields)):
            if yield_val != 0.0:  # Only include active metabolites
                monodK = self.monodKs[i]
                feeding_term_data["metDict"][met_name] = [yield_val, monodK]
        
        return feeding_term_data
    
    # ----------------------------------------------------------------------- #
    # JSON serialization
    # ----------------------------------------------------------------------- #
    
    def to_json(self, filename: str = None, full_model: bool = False, metabolome_obj: 'Metabolome' = None) -> str:
        """
        Convert the feeding term to JSON format.
        
        Parameters
        ----------
        filename : str, optional
            If provided, save the JSON to this file.
        full_model : bool, default False
            If True, returns a complete model structure ready for visualization.
            If False, returns just the feeding term data.
        metabolome_obj : Metabolome, optional
            Metabolome object to include in full model JSON. If not provided and full_model=True,
            will use the metabolome from which this feeding term was created.
            
        Returns
        -------
        str
            JSON string representation of the feeding term.
            
        Examples
        --------
        >>> # Create feeding term
        >>> feeding_term = FeedingTerm("glucose_consumer", {"glucose": [1.0, 0.5]}, metabolome)
        >>> 
        >>> # Get just the feeding term data
        >>> json_str = feeding_term.to_json()
        >>> 
        >>> # Get complete model structure with metabolome data
        >>> model_json = feeding_term.to_json(full_model=True, metabolome_obj=metabolome)
        >>> 
        >>> # Save to file
        >>> feeding_term.to_json("feeding_term.json")
        
        Notes
        -----
        The JSON structure follows the MicrobiomeGym model schema:
        - id: Feeding term identifier
        - metDict: Dictionary of metabolite consumption/production rates
        - metabolome_metabolites: List of metabolite names in the metabolome
        
        Schema compliance:
        - metDict format: {metabolite_name: [yield, monodK]}
        - Positive yields indicate consumption, negative yields indicate production
        - Monod K values are automatically set to 0 for produced metabolites
        - metabolome_metabolites list matches the metabolome structure
        
        Note: When full_model=True, you should provide metabolome_obj to ensure
        metabolite nodes are visible in visualizations.
        """
        import json
        
        # Create feeding term data according to schema
        feeding_term_data = {
            "id": self.id,
            "metDict": {},
            "metabolome_metabolites": self.metabolite_names.copy()
        }
        
        # Build metDict with only non-zero yields (active metabolites)
        for i, (met_name, yield_val) in enumerate(zip(self.metabolite_names, self.yields)):
            if yield_val != 0.0:  # Only include active metabolites
                monodK = self.monodKs[i]
                feeding_term_data["metDict"][met_name] = [yield_val, monodK]
        
        if full_model:
            # Use provided metabolome_obj or fall back to the one used to create this feeding term
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
                # Fallback: create minimal metabolite entries from feeding term data
                for met_name in self.metabolite_names:
                    metabolites_data[met_name] = {
                        "name": met_name,
                        "concentration": 1.0,  # Default concentration
                        "formula": {"C": 1, "H": 1, "O": 1},  # Default formula
                        "color": "#cccccc"  # Default color
                    }
            
            # Return complete model structure with actual metabolite data
            # Note: FeedingTerm doesn't have pH sensitivity parameters - those belong to Subpopulation
            model_data = {
                "version": "1.0.0",
                "metabolome": {
                    "metabolites": metabolites_data,  # Now includes actual metabolite data
                    "pH": {
                        "name": "pH Control",
                        "baseValue": 7.0,
                        "color": "#10b981",
                        "description": "Default pH control"
                    },
                    "temperature": 37.0,
                    "stirring": 1.0
                },
                "microbiome": {
                    "name": "default_community",
                    "bacteria": {
                        "default_species": {
                            "species": "default_species",
                            "color": "#ff4444",
                            "subpopulations": {
                                "default_subpop": {
                                    "name": "default_subpop",
                                    "count": 1.0,
                                    "species": "default_species",
                                    "mumax": 0.5,
                                    "feedingTerms": [feeding_term_data],
                                    "pHopt": 7.0,
                                    "pH_sensitivity_left": 2.0,  # Fixed: using new parameter names
                                    "pH_sensitivity_right": 2.0,  # Fixed: using new parameter names
                                    "Topt": 37.0,
                                    "tempSensitivity_left": 5.0,
                                    "tempSensitivity_right": 2.0,
                                    "state": "active",
                                    "color": "#ff4444"
                                }
                            },
                            "connections": {}
                        }
                    }
                }
            }
            json_str = json.dumps(model_data, indent=2)
        else:
            # Return just feeding term data
            json_str = json.dumps(feeding_term_data, indent=2)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(json_str)
        
        return json_str
