# -*- coding: utf-8 -*-
"""
kinetic_model.subpopulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines :class:`~kinetic_model.subpopulation.Subpopulation`, representing a
microbial subpopulation with its growth kinetics and metabolic strategy.

Example
-------
>>> from kinetic_model.metabolite import Metabolite
>>> from kinetic_model.metabolome import Metabolome
>>> from kinetic_model.feeding_term import FeedingTerm
>>> from kinetic_model.subpopulation import Subpopulation
>>>
>>> # Build a tiny metabolome
>>> glucose  = Metabolite("glucose", 10.0, {"C": 6, "H": 12, "O": 6})
>>> pyruvate = Metabolite("pyruvate", 5.0,  {"C": 3, "H": 4,  "O": 3})
>>> lactate  = Metabolite("lactate",  0.0,  {"C": 3, "H": 6,  "O": 3})
>>> metabolome = Metabolome([glucose, pyruvate, lactate])
>>>
>>> # Feeding term: consume glucose+pyruvate, produce lactate
>>> ft = FeedingTerm(
...     id="glc_pyr_to_lac",
...     metDict={
...         "glucose":  (1.0, 0.5),   # consume (yield>0, Monod K>0)
...         "pyruvate": (0.7, 0.3),   # consume
...         "lactate":  (-0.8, 0.0),  # produce (yield<0, Monod K=0)
...     },
...     metabolome=metabolome,
... )
>>>
>>> sp = Subpopulation(
...     name="xa", species="bh", count=1.0, mumax=0.8,
...     feedingTerms=[ft],
...     pHopt=7.0, pH_sensitivity_left=2.0, pH_sensitivity_right=2.0,
...     Topt=37.0, tempSensitivity_left=5.0, tempSensitivity_right=2.0,
...     state="active", color="#cf6f15",
... )
>>>
>>> c = metabolome.get_concentration()
>>> growth = sp.intrinsicGrowth(c)       # scalar in [0, mumax] after sensitivities
>>> rates  = sp.intrinsicMetabolism(c)   # vector aligned with metabolome order
"""


# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
from __future__ import annotations

import json
import numpy as np
from typing import List, Callable, Any, Dict

from .feeding_term import FeedingTerm


# --------------------------------------------------------------------------- #
# Public class
# --------------------------------------------------------------------------- #
class Subpopulation:
    """
    Represents a microbial subpopulation with specific growth characteristics.
    
    A Subpopulation defines a group of microbes with:
    - Growth rate parameters (mumax) - maximum growth rate under optimal conditions
    - pH sensitivity (pHopt, pH_sensitivity_left, pH_sensitivity_right) - optimal pH and asymmetric sensitivity using Gaussian distribution
    - Temperature sensitivity (Topt, tempSigma_left, tempSigma_right) - optimal temperature and asymmetric sensitivity
    - Multiple feeding terms for metabolic capabilities (OR relationships)
    - Population count and state management (active/inactive/dead)
    
    Parameters
    ----------
    name : str
        Name/identifier of the subpopulation
    count : float
        Concentration of the subpopulation (in cells/10**5)
    species : str
        Species name
    mumax : float
        Maximum growth rate
    feedingTerms : List[FeedingTerm]
        List of feeding term objects defining metabolic capabilities
    pHopt : float
        Optimal pH for growth
    pH_sensitivity_left : float
        pH sensitivity parameter for pH below optimal (gradual decline)
    pH_sensitivity_right : float
        pH sensitivity parameter for pH above optimal (sharp decline)
    Topt : float
        Optimal temperature for growth (in Celsius)
    tempSensitivity_left : float
        Temperature sensitivity parameter for temperatures below optimal (gradual decline)
    tempSensitivity_right : float
        Temperature sensitivity parameter for temperatures above optimal (sharp decline)
    state : str, optional
        One of three states: 'active', 'inactive' (pi positive), 'dead' (burst).
        Default is 'active'.
    color : str, optional
        Display color for visualization. Default is '#cf6f15'.
        
    Attributes
    ----------
    name : str
        Name/identifier of the subpopulation
    count : float
        Current count in cells/10**5 (always ≥ 0)
    species : str
        Species name
    mumax : float
        Maximum growth rate
    feedingTerms : List[FeedingTerm]
        List of feeding term objects defining metabolic capabilities
    pHopt : float
        Optimal pH for growth
    pH_sensitivity_left : float
        pH sensitivity parameter for pH below optimal (gradual decline)
    pH_sensitivity_right : float
        pH sensitivity parameter for pH above optimal (sharp decline)
    Topt : float
        Optimal temperature for growth (in Celsius)
    tempSensitivity_left : float
        Temperature sensitivity parameter for temperatures below optimal (gradual decline)
    tempSensitivity_right : float
        Temperature sensitivity parameter for temperatures above optimal (sharp decline)
    state : str
        Current state: 'active', 'inactive', or 'dead'
    color : str
        Display color for visualization
    pHSensitivity : Callable
        Function that calculates pH sensitivity factor
    tempSensitivity : Callable
        Function that calculates temperature sensitivity factor using asymmetric Gaussian
    intrinsicGrowth : Callable
        Function that calculates intrinsic growth rate from concentration vector
    intrinsicMetabolism : Callable
        Function that calculates intrinsic metabolism rates from concentration vector
        
    Raises
    ------
    ValueError
        If name is empty, count is negative, mumax is non-positive, 
        pHopt is non-positive, Topt is non-positive, sigma parameters are non-positive,
        or state is invalid
    TypeError
        If name is not a string, feedingTerms is not a list, or other type mismatches
        
    Notes
    -----
    * The subpopulation uses asymmetric Gaussian distribution for both pH and temperature sensitivity
    * Feeding terms use OR relationships (any can be satisfied for growth)
    * Count is automatically clamped to 0 if it would go negative
    * Active subpopulations must have positive mumax and at least one feeding term
    * Growth and metabolism methods now take concentration vectors directly instead of metabolome objects
    """
    
    def __init__(self, name: str, count: float, species: str, mumax: float, 
                 feedingTerms: List[FeedingTerm], pHopt: float, pH_sensitivity_left: float, pH_sensitivity_right: float,
                 Topt: float, tempSensitivity_left: float, tempSensitivity_right: float,
                 state: str = 'active', color: str = '#cf6f15'):
        """
        Initialize a Subpopulation.
        
        Parameters
        ----------
        name : str
            Name of the subpopulation
        count : float
            Concentration of the subpopulation (in cells/10**5)
        species : str
            Species name
        mumax : float
            Maximum growth rate
        feedingTerms : List[FeedingTerm]
            List of feeding term objects
        pHopt : float
            Optimal pH for growth
        pH_sensitivity_left : float
            pH sensitivity parameter for pH below optimal (gradual decline)
        pH_sensitivity_right : float
            pH sensitivity parameter for pH above optimal (sharp decline)
        Topt : float
            Optimal temperature for growth (in °C)
        tempSensitivity_left : float
            Temperature sensitivity parameter for temperatures below optimal (gradual decline)
        tempSensitivity_right : float
            Temperature sensitivity parameter for temperatures above optimal (sharp decline)
        state : str, optional
            One of three states: 'active', 'inactive' (pi positive), 'dead' (burst).
            Default is 'active'.
        color : str, optional
            Display color for visualization. Default is '#cf6f15'.
        """
        # Validate inputs
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not name.strip():
            raise ValueError("name cannot be empty")
        if not isinstance(count, (int, float)):
            raise TypeError("count must be a number")
        if count < 0:
            raise ValueError("count cannot be negative")
        if not isinstance(species, str):
            raise TypeError("species must be a string")
        if not isinstance(mumax, (int, float)):
            raise TypeError("mumax must be a number")
        if mumax < 0:
            raise ValueError("mumax cannot be negative")
        if state == 'active' and mumax <= 0:
            raise ValueError("active subpopulations must have positive mumax")
        if not isinstance(feedingTerms, list):
            raise TypeError("feedingTerms must be a list")
        if state == 'active' and not feedingTerms:
            raise ValueError("active subpopulations must have at least one feeding term")
        for term in feedingTerms:
            if not isinstance(term, FeedingTerm):
                raise TypeError("feedingTerms must contain FeedingTerm objects")
        if not isinstance(pHopt, (int, float)):
            raise TypeError("pHopt must be a number")
        if pHopt <= 0:
            raise ValueError("pHopt must be positive")
        if not isinstance(pH_sensitivity_left, (int, float)):
            raise TypeError("pH_sensitivity_left must be a number")
        if pH_sensitivity_left <= 0:
            raise ValueError("pH_sensitivity_left must be positive")
        if not isinstance(pH_sensitivity_right, (int, float)):
            raise TypeError("pH_sensitivity_right must be a number")
        if pH_sensitivity_right <= 0:
            raise ValueError("pH_sensitivity_right must be positive")
        if not isinstance(Topt, (int, float)):
            raise TypeError("Topt must be a number")
        if Topt <= 0:
            raise ValueError("Topt must be positive")
        if not isinstance(tempSensitivity_left, (int, float)):
            raise TypeError("tempSensitivity_left must be a number")
        if tempSensitivity_left <= 0:
            raise ValueError("tempSensitivity_left must be positive")
        if not isinstance(tempSensitivity_right, (int, float)):
            raise TypeError("tempSensitivity_right must be a number")
        if tempSensitivity_right <= 0:
            raise ValueError("tempSensitivity_right must be positive")
        if not isinstance(state, str):
            raise TypeError("state must be a string")
        if state not in ['active', 'inactive', 'dead']:
            raise ValueError("state must be one of: 'active', 'inactive', 'dead'")
        if not isinstance(color, str):
            raise TypeError("color must be a string")
            
        self.name = name
        self.species = species
        self.mumax = mumax
        self.state = state
        self.color = color
        
        # Initialize count using the property setter for validation
        self._count = 0.0  # Will be set by the property setter
        self.count = count
        
        self.feedingTerms = feedingTerms
        self.pHopt = pHopt
        self.pH_sensitivity_left = pH_sensitivity_left
        self.pH_sensitivity_right = pH_sensitivity_right
        self.Topt = Topt
        self.tempSensitivity_left = tempSensitivity_left
        self.tempSensitivity_right = tempSensitivity_right
        
        # Initialize computed properties
        self.pHSensitivity = self.__getpHSensitivity()
        self.tempSensitivity = self.__getTempSensitivity()
        self.intrinsicGrowth = self.__getIntrGrowth()
        self.intrinsicMetabolism = self.__getIntrMetabolism()
    
    # ----------------------------------------------------------------------- #
    # Count property (mutable)
    # ----------------------------------------------------------------------- #
    @property
    def count(self) -> float:
        """Current count in cells/10**5 (always ≥ 0)."""
        return self._count

    @count.setter
    def count(self, value: float) -> None:
        """Set a new count, clamp to 0, and validate."""
        try:
            val = float(value)
        except Exception as exc:
            raise TypeError(f"Count must be numeric, got {value!r}") from exc

        if val < 0.0 or val != val:  # also blocks NaN
            val = 0.0

        self._count = val
    
    # ----------------------------------------------------------------------- #
    # Private helper methods
    # ----------------------------------------------------------------------- #
    def __getpHSensitivity(self) -> Callable[[float], float]:
        """
        Create pH sensitivity function using asymmetric Gaussian distribution.
        
        Returns
        -------
        Callable[[float], float]
            Function that calculates pH sensitivity factor
        """
        def pHSensitivity(pH: float) -> float:
            if pH < self.pHopt:
                return np.exp(-((pH - self.pHopt) / self.pH_sensitivity_left)**2)
            else:
                return np.exp(-((pH - self.pHopt) / self.pH_sensitivity_right)**2)
        
        return pHSensitivity
    
    def __getTempSensitivity(self) -> Callable[[float], float]:
        """
        Create temperature sensitivity function using asymmetric Gaussian.
        
        Returns
        -------
        Callable[[float], float]
            Function that calculates temperature sensitivity factor
        """
        def tempSensitivity(T: float) -> float:
            if T < self.Topt:
                return np.exp(-((T - self.Topt) / self.tempSensitivity_left)**2)
            else:
                return np.exp(-((T - self.Topt) / self.tempSensitivity_right)**2)
        
        return tempSensitivity
    
    def __getIntrGrowth(self) -> Callable[[np.ndarray], float]:
        """
        Create intrinsic growth rate function that takes concentration vector.
        
        Returns
        -------
        Callable[[np.ndarray], float]
            Function that calculates intrinsic growth rate from concentration vector
        """
        def gr(concentrations: np.ndarray) -> float:
            growth = 0
            for fterm in self.feedingTerms:
                growth += fterm.intrinsicGrowth(concentrations)
            
            return self.mumax * self.count * growth
        
        return gr
    
    def __getIntrMetabolism(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create intrinsic metabolism function that takes concentration vector.
        
        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Function that calculates intrinsic metabolism rates from concentration vector
        """
        def metabolism(concentrations: np.ndarray) -> np.ndarray:
            metabV = np.zeros(len(concentrations))
            
            for fterm in self.feedingTerms:
                metabV += fterm.intrinsicMetabolism(concentrations)
            
            return self.mumax * self.count * metabV
        
        return metabolism
    
    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #
    def add(self, delta: float) -> None:
        """
        Add (or subtract if negative) to the current count.
        
        Parameters
        ----------
        delta : float
            Amount to add (positive) or subtract (negative) from current count.
            The result is clamped to 0 if it would go negative.
            
        Examples
        --------
        >>> subpop = Subpopulation("test", 10.0, "E. coli", 0.8, [], 7.0, 2.0, 2.0, 37.0, 5.0, 2.0)
        >>> subpop.add(5.0)    # 10 + 5 = 15 cells
        >>> subpop.count
        15.0
        >>> subpop.add(-8.0)   # 15 - 8 = 7 cells
        >>> subpop.count
        7.0
        >>> subpop.add(-10.0)  # 7 - 10 = -3, clamped to 0
        >>> subpop.count
        0.0
        """
        self.count = self.count + delta

    def update(self, new_count: float) -> None:
        """
        Replace the current count with a new value.
        
        Parameters
        ----------
        new_count : float
            New count value. Negative values are clamped to 0.
            
        Examples
        --------
        >>> subpop = Subpopulation("test", 10.0, "E. coli", 0.8, [], 7.0, 2.0, 2.0, 37.0, 5.0, 2.0)
        >>> subpop.update(15.0)  # Set to 15 cells
        >>> subpop.count
        15.0
        >>> subpop.update(-5.0)  # Try to set to -5, clamped to 0
        >>> subpop.count
        0.0
        """
        self.count = new_count

    def set_count(self, value: float) -> None:
        """
        Legacy alias for update() - kept for backward compatibility.
        
        Parameters
        ----------
        value : float
            New count value.
        """
        self.update(value)
    
    # ----------------------------------------------------------------------- #
    # JSON serialization methods
    # ----------------------------------------------------------------------- #
    def to_json(self, filename: str = None, full_model: bool = False) -> str:
        """
        Convert the subpopulation to JSON format.
        
        Parameters
        ----------
        filename : str, optional
            If provided, save the JSON to this file.
        full_model : bool, default False
            If True, returns a complete model structure ready for visualization.
            If False, returns just the subpopulation data.
            
        Returns
        -------
        str
            JSON string representation of the subpopulation.
            
        Examples
        --------
        >>> subpop = Subpopulation("test", 1.0, "E. coli", 0.5, [], 7.0, 2.0, 2.0, 37.0, 5.0, 2.0)
        >>> json_str = subpop.to_json()  # Just subpopulation data
        >>> model_json = subpop.to_json(full_model=True)  # Complete model structure
        >>> subpop.to_json("subpop.json")  # Save to file
        """
        # Create subpopulation data according to schema
        subpopulation_data = {
            "name": self.name,
            "count": self.count,
            "species": self.species,
            "mumax": self.mumax,
            "feedingTerms": [ft.get_data() for ft in self.feedingTerms],
            "pHopt": self.pHopt,
            "pH_sensitivity_left": self.pH_sensitivity_left,
            "pH_sensitivity_right": self.pH_sensitivity_right,
            "Topt": self.Topt,
            "tempSensitivity_left": self.tempSensitivity_left,
            "tempSensitivity_right": self.tempSensitivity_right,
            "state": self.state,
            "color": self.color
        }
        
        if full_model:
            # Build metabolites data from feeding terms
            metabolites_data = {}
            for ft in self.feedingTerms:
                # Get the metabolome from the feeding term
                metabolome = ft._metabolome
                if metabolome:
                    # Use actual metabolites from the metabolome
                    for met in metabolome._metabolite_dict.values():
                        if met.name not in metabolites_data:
                            metabolites_data[met.name] = {
                                "id": met.name,
                                "name": met.name,
                                "concentration": met.concentration,
                                "formula": met.formula,
                                "color": met.color
                            }
                else:
                    # Fallback: create metabolite entry with default values
                    for met_name in ft.metabolite_names:
                        if met_name not in metabolites_data:
                            metabolites_data[met_name] = {
                                "id": met_name,
                                "name": met_name,
                                "concentration": 1.0,  # Default concentration
                                "formula": {"C": 1, "H": 1, "O": 1},  # Default formula
                                "color": "#cccccc"  # Default color
                            }
            
            # Return complete model structure for integration
            model_data = {
                "version": "1.0.0",
                "metabolome": {
                    "metabolites": metabolites_data,
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
                            "species": self.species,
                            "color": self.color,
                            "subpopulations": {
                                self.name: subpopulation_data
                            },
                            "connections": {}
                        }
                    }
                }
            }
            json_str = json.dumps(model_data, indent=2)
        else:
            # Return just subpopulation data
            json_str = json.dumps(subpopulation_data, indent=2)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def get_complete_model(self) -> Dict[str, Any]:
        """
        Get complete model data for visualization and integration.
        
        Returns
        -------
        dict
            Complete model data including version, metabolome, and microbiome structure.
        """
        # Build metabolites data from feeding terms
        metabolites_data = {}
        for ft in self.feedingTerms:
            # Get the metabolome from the feeding term
            metabolome = ft._metabolome
            if metabolome:
                # Use actual metabolites from the metabolome
                for met in metabolome._metabolite_dict.values():
                    if met.name not in metabolites_data:
                        metabolites_data[met.name] = {
                            "id": met.name,
                            "name": met.name,
                            "concentration": met.concentration,
                            "formula": met.formula,
                            "color": met.color
                        }
            else:
                # Fallback: create metabolite entry with default values
                for met_name in ft.metabolite_names:
                    if met_name not in metabolites_data:
                        metabolites_data[met_name] = {
                            "id": met_name,
                            "name": met_name,
                            "concentration": 1.0,  # Default concentration
                            "formula": {"C": 1, "H": 1, "O": 1},  # Default formula
                            "color": "#cccccc"  # Default color
                        }
        
        return {
            "version": "1.0.0",
            "metabolome": {
                "metabolites": metabolites_data,
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
                        "species": self.species,
                        "color": self.color,
                        "subpopulations": {
                            self.name: {
                                "name": self.name,
                                "count": self.count,
                                "species": self.species,
                                "mumax": self.mumax,
                                "feedingTerms": [ft.get_data() for ft in self.feedingTerms],
                                "pHopt": self.pHopt,
                                "pH_sensitivity_left": self.pH_sensitivity_left,
                                "pH_sensitivity_right": self.pH_sensitivity_right,
                                "Topt": self.Topt,
                                "tempSensitivity_left": self.tempSensitivity_left,
                                "tempSensitivity_right": self.tempSensitivity_right,
                                "state": self.state,
                                "color": self.color
                            }
                        },
                        "connections": {}
                    }
                }
            }
        }
    
    # ----------------------------------------------------------------------- #
    # String representations
    # ----------------------------------------------------------------------- #
    def __repr__(self) -> str:
        """Return string representation of the Subpopulation."""
        return f"Subpopulation(name='{self.name}', species='{self.species}', count={self.count}, state='{self.state}')"
    
    def __str__(self) -> str:
        """Return string representation of the Subpopulation."""
        return f"Subpopulation {self.name} ({self.species}) - Count: {self.count}, State: {self.state}"


# --------------------------------------------------------------------------- #
# End of file
# --------------------------------------------------------------------------- #
