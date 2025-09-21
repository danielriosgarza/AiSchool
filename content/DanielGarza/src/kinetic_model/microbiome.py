# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
kinetic_model.microbiome
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines :class:`~kinetic_model.microbiome.Microbiome`, representing a
microbial community composed of multiple bacterial species.

Example
-------
>>> import numpy as np
>>> from kinetic_model.metabolite import Metabolite
>>> from kinetic_model.metabolome import Metabolome
>>> from kinetic_model.feeding_term import FeedingTerm
>>> from kinetic_model.subpopulation import Subpopulation
>>> from kinetic_model.bacteria import Bacteria, evaluate_transition_condition
>>> from kinetic_model.ph import pH
>>> from kinetic_model.stirring import Stirring
>>> from kinetic_model.temperature import Temperature
>>> from kinetic_model.environment import Environment
>>> from kinetic_model.microbiome import Microbiome
>>>
>>> glc = Metabolite("glucose", 10.0, {"C": 6, "H": 12, "O": 6})
>>> lac = Metabolite("lactate", 0.0, {"C": 3, "H": 6, "O": 3})
>>> pyr = Metabolite("pyruvate", 5.0, {"C": 3, "H": 4, "O": 3})
>>> met = Metabolome([glc, lac, pyr])
>>> ft = FeedingTerm("glc_pyr_to_lac", {"glucose": (1.0, 0.5), "pyruvate": (0.7, 0.3), "lactate": (-0.8, 0.0)}, met)
>>> sp = Subpopulation("xa", 1.0, "bh", 0.8, [ft], 7.0, 2.0, 2.0, 37.0, 5.0, 2.0, state="active", color="#cf6f15")
>>> cond = evaluate_transition_condition("lactate > 0.2", metabolome=met)
>>> bac = Bacteria("bh", {"xa": sp}, {"xa": [["xa", cond, 0.1]]}, color="#ff00aa", metabolome=met)
>>> env = Environment(pH(met, 7.0, {"glucose": -0.05}), Stirring(rate=1.0, base_std=0.0), Temperature(37.0))
>>> mic = Microbiome("toy", {"bh": bac})
>>> c = met.get_concentration()
>>> g = mic.growth(c, env); v = mic.metabolism(c, env); comp = mic.count()
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict

import numpy as np

from .metabolome import Metabolome
from .environment import Environment

# at top of the module
from .utils import json_safe

# Import plotly for visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import Bacteria only for type checking to avoid circular import at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # pragma: no cover
    from .bacteria import Bacteria



class Microbiome:
    """
    Represents a microbial community with multiple bacterial species.
    
    The Microbiome class manages a collection of bacterial species and their
    interactions within a shared environment. It provides methods for:
    - Tracking population composition across all species
    - Computing community-wide growth and metabolism rates from concentration vectors
    - Managing species interactions and environmental responses
    - JSON serialization compatible with the model-config schema
    
    The Microbiome class does not integrate or update the state of the system.
    Instead, it computes the instantaneous rates of:
    - Biomass changes across all bacterial species
    - Metabolite exchange rates for the entire community
    - Species interaction dynamics
    
    These rates are meant to be consumed by a Reactor or ODE integration class.
    
    Parameters
    ----------
    name : str
        Name/identifier of the microbial community
    bacteria : Dict[str, Bacteria]
        Dictionary mapping bacterial species names to Bacteria objects
    color : str, optional
        Display color for visualization. Default is '#2ecc71'.
        
    Attributes
    ----------
    name : str
        Name/identifier of the microbial community
    bacteria : Dict[str, Bacteria]
        Dictionary mapping bacterial species names to Bacteria objects
    color : str
        Display color for visualization
    composition : Dict[str, Dict[str, float]]
        Current composition of each bacterial species by state
        
    Raises
    ------
    ValueError
        If name is empty, bacteria is empty, or bacteria contains invalid objects
    TypeError
        If name is not a string, bacteria is not a dict, or other type mismatches
        
    Notes
    -----
    * Growth and metabolism methods now take concentration vectors directly instead of metabolome objects
    * The concentration vector should match the order of metabolites in the metabolome
    * All bacterial species must have compatible interfaces (count, growth, metabolism methods)
    """
    
    def __init__(self, name: str, bacteria: Dict[str, Any], color: str = '#2ecc71'):
        """
        Initialize a Microbiome.
        
        Parameters
        ----------
        name : str
            Name of the microbial community
        bacteria : Dict[str, Bacteria]
            Dictionary mapping bacterial species names to Bacteria objects
        color : str, optional
            Display color for visualization. Default is '#2ecc71'.
            
        Raises
        ------
        ValueError
            If name is empty, bacteria is empty, or bacteria contains invalid objects
        TypeError
            If name is not a string, bacteria is not a dict, or other type mismatches
        """
        # Validate inputs
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not name.strip():
            raise ValueError("name cannot be empty")
        if not isinstance(bacteria, dict):
            raise TypeError("bacteria must be a dictionary")
        if not bacteria:
            raise ValueError("bacteria cannot be empty")
        if not isinstance(color, str):
            raise TypeError("color must be a string")
        
        # Validate bacteria dictionary
        for species_name, bacteria_obj in bacteria.items():
            if not isinstance(species_name, str):
                raise TypeError("bacterial species names must be strings")
            if not species_name.strip():
                raise ValueError("bacterial species names cannot be empty")
            # Check if it has the required methods (for testing)
            if not (hasattr(bacteria_obj, 'count') and hasattr(bacteria_obj, 'growth') and hasattr(bacteria_obj, 'metabolism')):
                raise TypeError(f"bacteria[{species_name}] must be a Bacteria object or have count/growth/metabolism methods")
        
        # Check for duplicate species names
        if len(bacteria) != len(set(bacteria.keys())):
            raise ValueError("bacterial species names must be unique")
        
        self.name = name
        self.bacteria = bacteria
        self.color = color
    
    def count(self) -> Dict[str, Dict[str, float]]:
        """
        Get the composition of all bacterial species by state.
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping species names to their composition dictionaries.
            Each composition dict has keys 'active', 'inactive', 'dead' with
            corresponding population counts.
            
        Example
        -------
        >>> microbiome.count()
        {
            'E_coli': {'active': 5.2, 'inactive': 1.1, 'dead': 0.3},
            'B_subtilis': {'active': 3.8, 'inactive': 0.5, 'dead': 0.1}
        }
        """
        composition = {}
        for species_name, bacteria_obj in self.bacteria.items():
            composition[species_name] = bacteria_obj.count()
        return composition
    
    def growth(self, concentrations: np.ndarray, environment: Environment) -> Dict[str, Dict[str, float]]:
        """
        Compute growth rates for all bacterial species.
        
        Parameters
        ----------
        concentrations : np.ndarray
            Array of metabolite concentrations in the same order as the metabolome
        environment : Environment
            Environment object containing pH, temperature, and stirring information
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping species names to their growth rate dictionaries.
            Each growth dict maps subpopulation names to their growth rates.
            
        Example
        -------
        >>> concentrations = np.array([10.0, 5.0, 0.0])  # [glucose, oxygen, lactate]
        >>> microbiome.growth(concentrations, environment)
        {
            'E_coli': {
                'aerobic_strain': 0.15,
                'glucose_fermenter': 0.08,
                'inactive_strain': -0.02
            },
            'B_subtilis': {
                'active_strain': 0.12,
                'spore_strain': 0.01
            }
        }
        """
        # Compute current pH from metabolite concentrations
        current_pH = environment.pH.compute_pH(concentrations) if environment else 7.0
        
        growth = {}
        for species_name, bacteria_obj in self.bacteria.items():
            growth[species_name] = bacteria_obj.growth(concentrations, environment)
        return growth
    
    def metabolism(self, concentrations: np.ndarray, environment: Environment) -> np.ndarray:
        """
        Compute metabolism rates for the entire microbial community.
        
        Parameters
        ----------
        concentrations : np.ndarray
            Array of metabolite concentrations in the same order as the metabolome
        environment : Environment
            Environment object containing pH, temperature, and stirring information
            
        Returns
        -------
        np.ndarray
            Array of metabolism rates for all metabolites in the metabolome.
            Positive values indicate consumption, negative values indicate production.
            
        Example
        -------
        >>> concentrations = np.array([10.0, 5.0, 0.0])  # [glucose, oxygen, lactate]
        >>> microbiome.metabolism(concentrations, environment)
        array([ 0.25, -0.15,  0.08, -0.32,  0.05, -0.12,  0.03])
        """
        # Initialize metabolism array with zeros
        # Use the length of concentrations as the number of metabolites
        metV = np.zeros(len(concentrations))
        
        # Compute current pH from metabolite concentrations
        current_pH = environment.pH.compute_pH(concentrations) if environment else 7.0
        
        # Sum metabolism from all bacterial species
        for bacteria_obj in self.bacteria.values():
            species_metabolism = bacteria_obj.metabolism(concentrations, environment)
            if len(species_metabolism) == len(metV):
                metV += species_metabolism
            else:
                # Handle size mismatch by truncating or padding
                min_size = min(len(species_metabolism), len(metV))
                metV[:min_size] += species_metabolism[:min_size]
        
        return metV
    
    def to_json(self, filename: str | None = None, full_model: bool = False) -> str:
        """
        Convert the microbiome to JSON format.

        Parameters
        ----------
        filename : str, optional
            If provided, save the JSON to this file.
        full_model : bool, default False
            If True, returns a complete model structure ready for visualization.
            If False, returns just the microbiome data.

        Returns
        -------
        str
            JSON string representation of the microbiome.
        """
        # Create microbiome data according to schema
        microbiome_data = {
            "name": self.name,
            "color": self.color,
            "bacteria": {}
        }

        # Add bacteria data
        for species_name, bacteria_obj in self.bacteria.items():
            microbiome_data["bacteria"][species_name] = {
                "species": bacteria_obj.species,
                "color": bacteria_obj.color,
                "subpopulations": {},
                "connections": {}
            }

            # Add subpopulations
            for subpop_name, subpop in bacteria_obj.subpopulations.items():
                microbiome_data["bacteria"][species_name]["subpopulations"][subpop_name] = {
                    "name": subpop.name,
                    "count": subpop.count,
                    "species": subpop.species,
                    "mumax": subpop.mumax,
                    "feedingTerms": [ft.get_data() for ft in subpop.feedingTerms],
                    "pHopt": subpop.pHopt,
                    "pHalpha": subpop.pH_sensitivity_left,  # Using left sensitivity as alpha
                    "Topt": subpop.Topt,
                    "tempSensitivity_left": subpop.tempSensitivity_left,
                    "tempSensitivity_right": subpop.tempSensitivity_right,
                    "state": subpop.state,
                    "color": subpop.color,
                }

            # Add connections with correct subpopulation names
            if hasattr(bacteria_obj, "connections") and bacteria_obj.connections:
                for source_key, transitions in bacteria_obj.connections.items():
                    if source_key in bacteria_obj.subpopulations:
                        source_name = bacteria_obj.subpopulations[source_key].name
                        microbiome_data["bacteria"][species_name]["connections"][source_name] = []
                        for target_key, condition, rate in transitions:
                            if target_key in bacteria_obj.subpopulations:
                                target_name = bacteria_obj.subpopulations[target_key].name
                                microbiome_data["bacteria"][species_name]["connections"][source_name].append(
                                    [target_name, condition, rate]
                                )

        if full_model:
            # Build metabolites data from all bacteria
            metabolites_data: dict[str, dict] = {}
            for bacteria_obj in self.bacteria.values():
                for subpop in bacteria_obj.subpopulations.values():
                    for ft in subpop.feedingTerms:
                        metabolome = getattr(ft, "_metabolome", None)
                        if metabolome:
                            for met in metabolome._metabolite_dict.values():
                                if met.name not in metabolites_data:
                                    metabolites_data[met.name] = {
                                        "name": met.name,
                                        "concentration": met.concentration,
                                        "formula": met.formula,
                                        "color": met.color,
                                    }
                        else:
                            for met_name in ft.metabolite_names:
                                if met_name not in metabolites_data:
                                    metabolites_data[met_name] = {
                                        "name": met_name,
                                        "concentration": 1.0,              # default
                                        "formula": {"C": 1, "H": 1, "O": 1},  # default
                                        "color": "#cccccc",                 # default
                                    }

            # Complete model structure (environment defaults if not attached here)
            model_data = {
                "version": "1.0.0",
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
                "microbiome": microbiome_data,
            }
            json_str = json.dumps(json_safe(model_data), indent=2, ensure_ascii=False)
        else:
            # Just the microbiome data
            json_str = json.dumps(json_safe(microbiome_data), indent=2, ensure_ascii=False)

        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str

   
    def make_plot(self, title: str | None = None, cols: int = 3) -> go.Figure:
        """
        Subplots: one panel per bacteria species.
        - Bars = subpopulations (color = subpopulation color)
        - State via hatch patterns: Active='', Inactive='/', Dead='x'
        - Within each subplot, subpops ordered Active → Inactive → Dead (then name)
        - Shared Y axis; species names shown centered at the BOTTOM of each subplot
        """
        species = list(self.bacteria.keys())
        if not species:
            fig = go.Figure()
            fig.add_annotation(text="No data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            fig.update_layout(title="Microbiome Composition (by species)")
            return fig

        def patt(state: str) -> str:
            s = (state or "inactive").lower()
            return "" if s == "active" else ("/" if s == "inactive" else "x")

        state_rank = {"active": 0, "inactive": 1, "dead": 2}

        # ---- grid ----
        cols = max(1, int(cols))
        rows = (len(species) + cols - 1) // cols
        fig = make_subplots(
            rows=rows, cols=cols,
            shared_yaxes=True,
            horizontal_spacing=0.07, vertical_spacing=0.12,
        )

        max_y = 0.0

        # ---- add one subplot per species ----
        for idx, sp in enumerate(species, start=1):
            r = (idx - 1) // cols + 1
            c = (idx - 1) % cols + 1
            bac = self.bacteria[sp]

            # Order subpops: state priority then name
            ordered = sorted(
                bac.subpopulations.values(),
                key=lambda s: (state_rank.get((s.state or "inactive").lower(), 3), s.name.lower()),
            )

            x_labels, y_vals, patterns, colors, states = [], [], [], [], []
            for sub in ordered:
                x_labels.append(sub.name)
                y_vals.append(float(sub.count))
                patterns.append(patt(sub.state))
                colors.append(sub.color)
                states.append(sub.state or "inactive")
                max_y = max(max_y, float(sub.count))

            fig.add_trace(
                go.Bar(
                    x=x_labels, y=y_vals,
                    marker=dict(
                        color=colors,
                        line=dict(color="rgba(0,0,0,0.45)", width=0.5),
                        pattern=dict(shape=patterns, solidity=0.30, size=6),
                    ),
                    showlegend=False,
                    customdata=states,
                    hovertemplate=(
                        f"<b>Species:</b> {sp}<br>"
                        "<b>Subpopulation:</b> %{x}<br>"
                        "<b>State:</b> %{customdata}<br>"
                        "<b>Count:</b> %{y:.3g}<extra></extra>"
                    ),
                ),
                row=r, col=c,
            )

            # No x-axis title; just tilt tick labels slightly
            fig.update_xaxes(tickangle=-20, row=r, col=c)

        # ---- add compact state legend (legend-only swatches) ----
        # Anchor them to the first subplot so they show in the main legend area.
        fig.add_trace(go.Bar(x=[None], y=[None], name="Active",  visible="legendonly",
                            marker=dict(color="#888", pattern=dict(shape="",  solidity=0.4, size=6)),
                            hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Bar(x=[None], y=[None], name="Inactive", visible="legendonly",
                            marker=dict(color="#888", pattern=dict(shape="/", solidity=0.4, size=6)),
                            hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Bar(x=[None], y=[None], name="Dead",    visible="legendonly",
                            marker=dict(color="#888", pattern=dict(shape="x", solidity=0.4, size=6)),
                            hoverinfo="skip"), row=1, col=1)

        # ---- layout ----
        t = title or "Microbiome Composition (by species)"
        headroom = 0.12 if max_y == 0 else 0.12 * max_y
        fig.update_yaxes(title_text="Population Count", range=[0, max_y + headroom], row=1, col=1)
        fig.update_layout(
            title=dict(text=t, x=0.5, xanchor="center", font=dict(size=18)),
            template="plotly_white",
            bargap=0.25,
            height=max(420, rows * 320),
            margin=dict(l=60, r=40, t=90, b=110),  # extra bottom space for species labels
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="center", x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.15)", borderwidth=1,
            ),
        )

        # ---- species names centered at the BOTTOM of each subplot ----
        # Use each subplot's x-domain to place a paper-referenced annotation at its center.
        for idx, sp in enumerate(species, start=1):
            ax_key = f"xaxis{idx}" if idx > 1 else "xaxis"
            if ax_key in fig.layout and "domain" in fig.layout[ax_key]:
                x0, x1 = fig.layout[ax_key]["domain"]
                x_center = (x0 + x1) / 2.0
                fig.add_annotation(
                    x=x_center, y=-0.12, xref="paper", yref="paper",
                    text=sp, showarrow=False,
                    font=dict(size=14),
                    xanchor="center", yanchor="top",
                )

        return fig










    def __repr__(self) -> str:
        """Return a detailed string representation of the Microbiome."""
        return f"Microbiome(name='{self.name}', species_count={len(self.bacteria)}, color='{self.color}')"
    
    def __str__(self) -> str:
        """Return a concise string representation of the Microbiome."""
        return f"Microbiome('{self.name}' with {len(self.bacteria)} species)"
