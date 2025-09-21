# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
kinetic_model.bacteria
~~~~~~~~~~~~~~~~~~~~~~~~~

Defines :class:`~kinetic_model.bacteria.Bacteria`, which represents
a bacterial species composed of interconnected subpopulations.

Example
-------
>>> import numpy as np
>>> from kinetic_model.metabolite import Metabolite
>>> from kinetic_model.metabolome import Metabolome
>>> from kinetic_model.feeding_term import FeedingTerm
>>> from kinetic_model.subpopulation import Subpopulation
>>> from kinetic_model.ph import pH
>>> from kinetic_model.stirring import Stirring
>>> from kinetic_model.temperature import Temperature
>>> from kinetic_model.environment import Environment
>>> from kinetic_model.bacteria import Bacteria, evaluate_transition_condition
>>>
>>> # Tiny metabolome
>>> glucose  = Metabolite("glucose", 10.0, {"C": 6, "H": 12, "O": 6})
>>> lactate  = Metabolite("lactate",  0.0, {"C": 3, "H": 6,  "O": 3})
>>> pyruvate = Metabolite("pyruvate", 5.0, {"C": 3, "H": 4,  "O": 3})
>>> met = Metabolome([glucose, lactate, pyruvate])
>>>
>>> # One feeding term
>>> ft = FeedingTerm("glc_pyr_to_lac", {
...     "glucose":  (1.0, 0.5),
...     "pyruvate": (0.7, 0.3),
...     "lactate":  (-0.8, 0.0),
... }, metabolome=met)
>>>
>>> sp = Subpopulation("xa", 1.0, "bh", 0.8, [ft],
...                    7.0, 2.0, 2.0, 37.0, 5.0, 2.0,
...                    state="active", color="#cf6f15")
>>>
>>> env = Environment(pH(met, 7.0, {"glucose": -0.1}),
...                   Stirring(rate=0.9, base_std=0.05),
...                   Temperature(37.0))
>>>
>>> cond = evaluate_transition_condition("lactate > 0.2", metabolome=met)
>>> connections = {"xa": [["xa", cond, 0.1]]}
>>>
>>> bac = Bacteria("example_species", {"xa": sp}, connections, color="#54f542", metabolome=met)
>>> c = met.get_concentration()
>>> growth = bac.growth(c, env)
>>> fluxes = bac.metabolism(c, env)
"""


import numpy as np
import json
from typing import Dict, List, Callable, Any, Tuple
from collections import defaultdict

from .subpopulation import Subpopulation
from .environment import Environment
from .utils import json_safe



def evaluate_transition_condition(expression: str, metabolome=None):
    """
    Create a transition condition function from a string expression.
    
    This function takes a mathematical expression as a string and returns a callable
    function that can be evaluated with concentrations and environment variables.
    
    If a metabolome is provided, metabolite names in the expression will be automatically
    replaced with their corresponding position in the concentration vector.
    
    Parameters
    ----------
    expression : str
        Mathematical expression as a string. Can use metabolite names (e.g., "glucose > 5.0")
        or direct concentration indices (e.g., "concentrations[0] > 5.0").
        Special cases:
        - Empty string "" or whitespace-only: returns 1.0 (always transition)
        - Two double quotes '""' (JSON empty condition): returns 1.0 (always transition)
        - Valid mathematical expression: evaluated and returned as float
        
    Examples
    --------
    >>> # Create condition function with metabolite names
    >>> condition_func = evaluate_transition_condition("glucose > 5.0", metabolome)
    >>> result = condition_func(environment, concentrations)
    
    >>> # Create condition function with direct indices (backward compatible)
    >>> condition_func = evaluate_transition_condition("concentrations[0] > 5.0")
    >>> result = condition_func(environment, concentrations)
    
    >>> # Complex condition with metabolite names
    >>> condition_func = evaluate_transition_condition("glucose < 0.1 and oxygen > 0.5", metabolome)
    >>> result = condition_func(environment, concentrations)
    
    >>> # Expression with environment attributes and metabolite names
    >>> condition_func = evaluate_transition_condition("(environment.pH > 3) and (environment.temperature == 34) * lactate", metabolome)
    >>> result = condition_func(environment, concentrations)
    
    >>> # Empty string (always transition)
    >>> condition_func = evaluate_transition_condition("")
    >>> result = condition_func(environment, concentrations)  # Always returns 1.0
    """
    # Handle special cases first
    if not expression or expression.strip() == "":
        # Empty string means always transition (return 1.0)
        def transition_function(environment: Environment, concentrations: np.ndarray) -> float:
            return 1.0
        return transition_function
    
    # Handle JSON empty string literal case
    if expression == '""' or expression == "''":
        def transition_function(environment: Environment, concentrations: np.ndarray) -> float:
            return 1.0
        return transition_function
    
    # Preprocess expression if metabolome is provided
    processed_expression = expression
    
    if metabolome is not None:
        # Create mapping from metabolite names to their indices
        metabolite_to_index = {name: i for i, name in enumerate(metabolome.metabolites)}
        
        # Replace metabolite names with concentrations[index]
        for metabolite_name, index in metabolite_to_index.items():
            # Use word boundaries to avoid partial replacements
            # This ensures "glucose" doesn't match "glucose_consumer"
            processed_expression = processed_expression.replace(
                metabolite_name, f"concentrations[{index}]"
            )
    
    def transition_function(environment: Environment, concentrations: np.ndarray) -> float:
        """Evaluate the condition expression with current environment and concentrations."""
        try:
            result = eval(processed_expression)
            return float(result)
        except Exception as e:
            # If evaluation fails, return 0.0 (no transition)
            print(f"Warning: Could not evaluate transition condition '{processed_expression}' (original: '{expression}'): {e}")
            return 0.0
    
    return transition_function


class Bacteria:
    """
    Represents a bacterial species with multiple interconnected subpopulations.
    
    A Bacteria defines a species containing:
    - Multiple subpopulations with different metabolic capabilities
    - Interconnections between subpopulations with conditional transitions
    - Growth and metabolism calculations considering pH and temperature sensitivity
    - Population composition tracking (active/inactive/dead states)
    
    The class computes instantaneous rates of biomass change and metabolite exchange,
    which are meant to be consumed by a Reactor or ODE integration class.
    
    Parameters
    ----------
    species : str
        Name/identifier of the bacterial species
    subpopulations : Dict[str, Subpopulation]
        Dictionary mapping subpopulation names to Subpopulation objects
    connections : Dict[str, List[List]]
        Dictionary mapping subpopulation names to lists of connection tuples.
        Each connection is [target_subpop, condition_function, transition_rate]
        where:
        - target_subpop: name of target subpopulation
        - condition_function: function that takes environment and returns float
          OR a string expression for conditional transitions
        - transition_rate: rate of transition between subpopulations
    color : str, optional
        Display color for visualization. Default is '#54f542'.
    metabolome : Metabolome, optional
        Metabolome object containing the ordered list of metabolites.
        If provided, metabolite names in transition conditions will be automatically
        replaced with their corresponding position in the concentration vector.
        This allows writing conditions like "glucose < 0.1" instead of "concentrations[0] < 0.1".
        
    Attributes
    ----------
    species : str
        Name/identifier of the bacterial species
    subpopulations : Dict[str, Subpopulation]
        Dictionary mapping subpopulation names to Subpopulation objects
    connections : Dict[str, List[List]]
        Dictionary mapping subpopulation names to lists of connection tuples
    color : str
        Display color for visualization
    metabolome : Metabolome or None
        Metabolome object for metabolite name resolution in transition conditions
    composition : Dict[str, float]
        Current composition of the bacterial species by state
        
    Examples
    --------
    >>> # Create bacteria with metabolite name support in transitions
    >>> bacteria = Bacteria(
    ...     species="E. coli",
    ...     subpopulations=subpopulations,
    ...     connections={
    ...         "aerobic": [["anaerobic", "glucose < 0.1 and oxygen < 0.5", 0.2]],
    ...         "anaerobic": [["inactive", "glucose < 0.1", 0.3]],
    ...         "inactive": [["aerobic", "glucose > 1.0 and oxygen > 0.5", 0.1]]
    ...     },
    ...     metabolome=metabolome  # Required for metabolite name resolution
    ... )
    >>> 
    >>> # Backward compatible: still works with concentration indices
    >>> bacteria_old = Bacteria(
    ...     species="E. coli",
    ...     subpopulations=subpopulations,
    ...     connections={
    ...         "aerobic": [["anaerobic", "concentrations[0] < 0.1 and concentrations[1] < 0.5", 0.2]]
    ...     }
    ...     # No metabolome needed for index-based expressions
    ... )
    """
    
    def __init__(self, species: str, subpopulations: Dict[str, Subpopulation], 
                 connections: Dict[str, List[List]], color: str = '#54f542', metabolome=None):
        """
        Initialize a Bacteria instance.
        
        Parameters
        ----------
        species : str
            Name of the bacterial species
        subpopulations : Dict[str, Subpopulation]
            Dictionary mapping subpopulation names to Subpopulation objects
        connections : Dict[str, List[List]]
            Dictionary mapping subpopulation names to lists of connection tuples.
            Each connection is [target_subpop, condition_function, transition_rate]
            where:
            - target_subpop: name of target subpopulation
            - condition_function: function that takes environment and returns float
              OR a string expression for conditional transitions
            - transition_rate: rate of transition between subpopulations
        color : str, optional
            Display color for visualization. Default is '#54f542'.
        metabolome : Metabolome, optional
            Metabolome object containing the ordered list of metabolites.
            If provided, metabolite names in transition conditions will be automatically
            replaced with their corresponding position in the concentration vector.
            This allows writing conditions like "glucose < 0.1" instead of "concentrations[0] < 0.1".
            
        Raises
        ------
        ValueError
            If species is empty, subpopulations is empty, or connections contain invalid references
        TypeError
            If species is not a string, subpopulations is not a dict, or other type mismatches
        """
        # Validate inputs
        if not isinstance(species, str):
            raise TypeError("species must be a string")
        if not species.strip():
            raise ValueError("species cannot be empty")
        if not isinstance(subpopulations, dict):
            raise TypeError("subpopulations must be a dictionary")
        if not subpopulations:
            raise ValueError("subpopulations cannot be empty")
        if not isinstance(connections, dict):
            raise TypeError("connections must be a dictionary")
        if not isinstance(color, str):
            raise TypeError("color must be a string")
        
        # Validate metabolome if provided
        if metabolome is not None:
            from .metabolome import Metabolome
            if not isinstance(metabolome, Metabolome):
                raise TypeError("metabolome must be a Metabolome instance")
        
        # Validate subpopulations
        for name, subpop in subpopulations.items():
            if not isinstance(name, str):
                raise TypeError("subpopulation names must be strings")
            if not isinstance(subpop, Subpopulation):
                raise TypeError("subpopulations must contain Subpopulation objects")
        
        # Validate connections
        for source_name, connection_list in connections.items():
            if not isinstance(source_name, str):
                raise TypeError("connection source names must be strings")
            if source_name not in subpopulations:
                raise ValueError(f"connection source '{source_name}' not found in subpopulations")
            if not isinstance(connection_list, list):
                raise TypeError(f"connections for '{source_name}' must be a list")
            
            for connection in connection_list:
                if not isinstance(connection, list) or len(connection) != 3:
                    raise ValueError(f"each connection must be a list of 3 elements: [target, condition, rate]")
                
                target_name, condition_func, transition_rate = connection
                
                if not isinstance(target_name, str):
                    raise TypeError("connection target names must be strings")
                if target_name not in subpopulations:
                    raise ValueError(f"connection target '{target_name}' not found in subpopulations")
                
                # Validate condition function (can be callable or string)
                if isinstance(condition_func, str):
                    # String expression - check if it's empty
                    if condition_func.strip() == "" or condition_func == '""' or condition_func == "''":
                        # Empty string is valid - means always transition
                        pass
                    else:
                        # Valid expression - let it pass through
                        pass
                elif not callable(condition_func):
                    raise TypeError(f"Connection condition must be a callable function or string expression, got {type(condition_func)}")
                
                if not isinstance(transition_rate, (int, float)):
                    raise TypeError("transition rate must be a number")
                if transition_rate < 0:
                    raise ValueError("transition rate cannot be negative")
        
        self.species = species
        self.subpopulations = subpopulations
        self.connections = connections
        self.color = color
        self.metabolome = metabolome
        self.composition = {'active': 0.0, 'inactive': 0.0, 'dead': 0.0}
    

    
    def count(self) -> Dict[str, float]:
        """
        Calculate the current composition of the bacterial species by state.
        
        Returns
        -------
        Dict[str, float]
            Dictionary with counts for each state: 'active', 'inactive', 'dead'
        """
        self.composition = {'active': 0.0, 'inactive': 0.0, 'dead': 0.0}
        
        for subpop_name in self.subpopulations:
            subpop = self.subpopulations[subpop_name]
            self.composition[subpop.state] += subpop.count
        
        return self.composition
    
    def growth(self, concentrations: np.ndarray, environment: Environment) -> Dict[str, float]:
        """
        Calculate instantaneous growth rates for all subpopulations considering connections.
        
        This method computes the rate of biomass change for each subpopulation:
        1. Intrinsic growth for each subpopulation
        2. pH and temperature sensitivity effects from environment
        3. Transitions between subpopulations based on conditions
        
        Note: This method returns rates, not integrated values. The rates should be
        integrated by a Reactor or ODE solver to update the actual biomass.
        
        Parameters
        ----------
        concentrations : np.ndarray
            Array of metabolite concentrations
        environment : Environment
            Environment object containing pH and temperature information
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping subpopulation names to their instantaneous growth rates
        """
        growth = {subpop_name: 0.0 for subpop_name in self.subpopulations}
        
        for subpop_name in self.subpopulations:
            subpop = self.subpopulations[subpop_name]
            
            # Calculate intrinsic growth with pH and temperature sensitivity
            intrinsic_growth = subpop.intrinsicGrowth(concentrations)
            current_ph = environment.pH.compute_pH(concentrations)
            ph_sensitivity = subpop.pHSensitivity(current_ph)
            temp_sensitivity = subpop.tempSensitivity(environment.temperature.temperature)
            growth[subpop_name] += intrinsic_growth * ph_sensitivity * temp_sensitivity
            
            # Handle connections/transitions
            if subpop_name in self.connections:
                for connection in self.connections[subpop_name]:
                    target_name, condition_func, transition_rate = connection
                    
                    # Get transition probability from condition function
                    if isinstance(condition_func, str):
                        # Create condition function from string expression
                        condition_function = evaluate_transition_condition(condition_func, self.metabolome)
                        transition_probability = condition_function(environment, concentrations)
                    else:
                        # Assume it's already a callable function
                        transition_probability = condition_func(environment, concentrations)
                    
                    # Apply transition based on probability
                    transition_amount = subpop.count * transition_rate * transition_probability
                    # Add to target subpopulation
                    growth[target_name] += transition_amount
                    # Subtract from source subpopulation
                    growth[subpop_name] -= transition_amount
        
        return growth
    
    def metabolism(self, concentrations: np.ndarray, environment: Environment) -> np.ndarray:
        """
        Calculate instantaneous metabolite exchange rates for all subpopulations.
        
        This method computes the rate of metabolite consumption/production for each
        metabolite based on the metabolism of all subpopulations, considering pH and
        temperature sensitivity effects from environment.
        
        Note: This method returns rates, not integrated values. The rates should be
        integrated by a Reactor or ODE solver to update the actual metabolite concentrations.
        
        Parameters
        ----------
        concentrations : np.ndarray
            Array of metabolite concentrations
        environment : Environment
            Environment object containing pH and temperature information
            
        Returns
        -------
        np.ndarray
            Array of instantaneous metabolite exchange rates for each metabolite
        """
        metV = np.zeros(len(concentrations))
        
        for subpop_name in self.subpopulations:
            subpop = self.subpopulations[subpop_name]
            
            # Calculate metabolism with pH and temperature sensitivity
            intrinsic_metabolism = subpop.intrinsicMetabolism(concentrations)
            current_ph = environment.pH.compute_pH(concentrations)
            ph_sensitivity = subpop.pHSensitivity(current_ph)
            temp_sensitivity = subpop.tempSensitivity(environment.temperature.temperature)
            metV += intrinsic_metabolism * ph_sensitivity * temp_sensitivity
        
        return metV
    
    def to_json(self, filename: str | None = None, full_model: bool = False) -> str:
        """
        Convert the bacteria to JSON format compatible with the model-config schema.

        Parameters
        ----------
        filename : str, optional
            If provided, save the JSON to this file.
        full_model : bool, default False
            If True, returns a complete model structure ready for visualization.
            If False, returns just the bacteria data.

        Returns
        -------
        str
            JSON string representation of the bacteria.
        """
        # Create bacteria data according to schema
        bacteria_data = {
            "species": self.species,
            "color": self.color,
            "subpopulations": {},
            "connections": {},
        }

        # Add subpopulation data
        for name, subpop in self.subpopulations.items():
            # Get the subpopulation JSON as dict and normalize id/name to the mapping key
            subpop_json = json.loads(subpop.to_json())
            subpop_json["name"] = name
            subpop_json["id"] = name
            bacteria_data["subpopulations"][name] = subpop_json

        # Add connections in schema-compatible format: [target, condition, rate]
        # Note: These are edges between existing subpopulations (no new nodes).
        for source_key, transitions in (self.connections or {}).items():
            if source_key not in self.subpopulations:
                # skip edges from unknown sources
                continue
            rows = []
            for target_key, condition_obj, transition_rate in transitions:
                if target_key not in self.subpopulations:
                    # skip edges to unknown targets
                    continue

                # Normalize condition:
                # - keep strings as-is
                # - replace callables with a stable placeholder (so JSON is deterministic)
                if isinstance(condition_obj, str):
                    condition_expr = condition_obj
                elif callable(condition_obj):
                    condition_expr = "<callable_condition>"
                else:
                    condition_expr = str(condition_obj)

                rows.append([target_key, condition_expr, float(transition_rate)])
            if rows:
                bacteria_data["connections"][source_key] = rows

        if full_model:
            # Build metabolites data from subpopulations for complete model
            metabolites_data: dict[str, dict] = {}
            for subpop in self.subpopulations.values():
                for ft in subpop.feedingTerms:
                    metabolome = getattr(ft, "_metabolome", None)
                    if metabolome is not None:
                        for met in getattr(metabolome, "_metabolite_dict", {}).values():
                            name = getattr(met, "name", None)
                            if not name or name in metabolites_data:
                                continue
                            metabolites_data[name] = {
                                "id": name,
                                "name": name,
                                "concentration": met.concentration,
                                "formula": met.formula,
                                "color": met.color,
                            }
                    else:
                        for met_name in getattr(ft, "metabolite_names", []):
                            if met_name in metabolites_data:
                                continue
                            metabolites_data[met_name] = {
                                "id": met_name,
                                "name": met_name,
                                "concentration": 1.0,              # default
                                "formula": {"C": 1, "H": 1, "O": 1},  # default
                                "color": "#cccccc",                 # default
                            }

            model_data = {
                "version": "1.0.0",
                "metadata": {
                    "name": f"{self.species} Community",
                    "description": (
                        f"Bacterial community with {len(self.subpopulations)} subpopulations "
                        "and transition dynamics"
                    ),
                    "author": "MicrobiomeGym",
                    "created": "2024-01-01T00:00:00Z",
                    "tags": ["bacteria", "transitions", "metabolism"],
                },
                "metabolome": {
                    "metabolites": list(metabolites_data.values()),
                    "pH": {
                        "name": "pH Control",
                        "baseValue": 7.0,
                        "color": "#10b981",
                        "description": "Default pH control",
                    },
                    "temperature": 37.0,
                    "stirring": 1.0,
                },
                "microbiome": {
                    "name": f"{self.species.lower()}_community",
                    "color": self.color,
                    "bacteria": {self.species: bacteria_data},
                },
            }
            json_str = json.dumps(json_safe(model_data), indent=2, ensure_ascii=False)
        else:
            json_str = json.dumps(json_safe(bacteria_data), indent=2, ensure_ascii=False)

        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str
    
    def to_visualization_json(self, filename: str = None) -> str:
        """
        Convert the bacteria to JSON format specifically designed for visualization systems.
        
        This method creates a structure where transitions are explicitly represented as edges
        between subpopulations, ensuring visualization systems treat them as connections
        rather than as separate nodes.
        
        Parameters
        ----------
        filename : str, optional
            If provided, save the JSON to this file
            
        Returns
        -------
        str
            JSON string representation optimized for visualization
        """
        # Create visualization-optimized structure
        viz_data = {
            "version": "1.0.0",
            "metadata": {
                "name": f"{self.species} Visualization",
                "description": f"Bacterial community visualization with {len(self.subpopulations)} subpopulations",
                "type": "bacteria_network"
            },
            "nodes": [],
            "edges": []
        }
        
        # Add subpopulation nodes
        for name, subpop in self.subpopulations.items():
            viz_data["nodes"].append({
                "id": name,
                "label": name,
                "type": "subpopulation",
                "data": {
                    "species": subpop.species,
                    "state": subpop.state,
                    "count": subpop.count,
                    "mumax": subpop.mumax,
                    "color": subpop.color
                }
            })
        
        # Add transition edges
        for source_name, connection_list in self.connections.items():
            for connection in connection_list:
                target_name, condition_func, transition_rate = connection
                
                # Handle different types of condition functions
                if isinstance(condition_func, str):
                    condition_expression = condition_func
                elif callable(condition_func):
                    condition_expression = str(condition_func)
                else:
                    condition_expression = str(condition_func)
                
                # Create edge with unique ID
                edge_id = f"{source_name}_to_{target_name}_{hash(condition_expression) % 10000}"
                
                viz_data["edges"].append({
                    "id": edge_id,
                    "source": source_name,
                    "target": target_name,
                    "label": f"{condition_expression} (rate: {transition_rate})",
                    "type": "transition",
                    "data": {
                        "condition": condition_expression,
                        "rate": transition_rate,
                        "source": source_name,
                        "target": target_name
                    }
                })
        
        # Convert to JSON
        json_str = json.dumps(viz_data, indent=2)
        
        # Save to file if filename provided
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        return json_str
    
    def __repr__(self) -> str:
        """Return string representation of the Bacteria."""
        return f"Bacteria(species='{self.species}', subpopulations={len(self.subpopulations)}, connections={len(self.connections)})"
    
    def __str__(self) -> str:
        """Return string representation of the Bacteria."""
        composition = self.count()
        return f"Bacteria {self.species} - Subpopulations: {len(self.subpopulations)}, Composition: {composition}"

