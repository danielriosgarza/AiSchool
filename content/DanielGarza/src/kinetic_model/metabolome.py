# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
kinetic_model.metabolome
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines :class:`~kinetic_model.metabolome.Metabolome`, which manages
a collection of metabolites.

A **Metabolome** instance

* maintains a collection of `Metabolite` objects
* provides convenient methods to update multiple metabolite concentrations at once
* offers interactive plotting with Plotly
* supports JSON serialization for data export and model sharing

Example
-------
>>> from kinetic_model.metabolite import Metabolite
>>> from kinetic_model.metabolome import Metabolome
>>> glucose = Metabolite("glucose", 10.0, {'C': 6, 'H': 12, 'O': 6})
>>> lactate = Metabolite("lactate", 5.0, {'C': 3, 'H': 6, 'O': 3})
>>> metabolome = Metabolome(metabolites=[glucose, lactate])
>>> metabolome.get_concentration()
array([10.,  5.])
>>> metabolome.add({'glucose': -2.0, 'lactate': 1.0})
>>> metabolome.get_concentration()
array([8., 6.])
>>> # metabolome.make_plot()  # Optional: interactive plot
>>> json_str = metabolome.to_json()            # Export to JSON
>>> model_json = metabolome.to_json(full_model=True)  # Complete model structure
"""


# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
from __future__ import annotations

from typing import Dict, List, Optional
import json
import numpy as np
import plotly.graph_objects as go

from .metabolite import Metabolite

# --------------------------------------------------------------------------- #
# Type aliases
# --------------------------------------------------------------------------- #
ConcentrationDict = Dict[str, float]

# --------------------------------------------------------------------------- #
# Public class
# --------------------------------------------------------------------------- #
class Metabolome:
    """
    Manages a collection of metabolites.

    Parameters
    ----------
    metabolites : List[Metabolite]
        List of metabolite objects to include in the metabolome.

    Notes
    -----
    * The metabolome maintains a sorted list of metabolite names for consistent
      ordering in concentration arrays.
    * The `get_concentration()` method returns a numpy array with concentrations
      in the same order as `self.metabolites`.
    * The `add()` method adds (or subtracts if negative) to metabolite concentrations.
    * The `update()` method replaces metabolite concentrations with new values.
    * The `make_plot()` method creates an interactive Plotly visualization.

    Attributes
    ----------
    metabolites : List[str]
        Sorted list of metabolite names.
    nmets : int
        Number of metabolites.
    """

    def __init__(
        self,
        metabolites: List[Metabolite]
    ) -> None:
        # Validate inputs
        if not isinstance(metabolites, list) or not metabolites:
            raise ValueError("metabolites must be a non-empty list")
        
        if not all(isinstance(m, Metabolite) for m in metabolites):
            raise TypeError("All metabolites must be Metabolite instances")
        
        # Store metabolites in a dictionary for fast lookup
        self._metabolite_dict = {met.name: met for met in metabolites}
        
        # Create sorted list of metabolite names for consistent ordering
        self.metabolites = sorted(self._metabolite_dict.keys())
        self.nmets = len(self.metabolites)

    # ----------------------------------------------------------------------- #
    # Concentration management
    # ----------------------------------------------------------------------- #
    def get_concentration(self) -> np.ndarray:
        """
        Get current metabolite concentrations as a numpy array.
        
        Returns
        -------
        np.ndarray
            Array of concentrations in the same order as `self.metabolites`.
        """
        return np.array([self._metabolite_dict[name].concentration 
                        for name in self.metabolites])

    def add(
        self,
        concentration_dict: ConcentrationDict
    ) -> None:
        """
        Add (or subtract if negative) to metabolite concentrations.
        
        Parameters
        ----------
        concentration_dict : ConcentrationDict
            Dictionary mapping metabolite names to amounts to add/subtract.
            Positive values add to current concentration, negative values subtract.
            Results are clamped to 0 if they would go negative.
            
        Examples
        --------
        >>> metabolome.add({'glucose': 5.0, 'lactate': -2.0})
        # Adds 5 mM to glucose, subtracts 2 mM from lactate
        """
        # Validate input
        if not isinstance(concentration_dict, dict):
            raise TypeError("concentration_dict must be a dictionary")
        
        # Check that all metabolite names exist
        unknown_metabolites = set(concentration_dict.keys()) - set(self.metabolites)
        if unknown_metabolites:
            raise ValueError(f"Unknown metabolites: {unknown_metabolites}")
        
        # Add to concentrations
        for name, value in concentration_dict.items():
            metabolite = self._metabolite_dict[name]
            metabolite.add(value)

    def update(
        self,
        concentration_dict: ConcentrationDict
    ) -> None:
        """
        Replace metabolite concentrations with new values.
        
        Parameters
        ----------
        concentration_dict : ConcentrationDict
            Dictionary mapping metabolite names to new concentration values.
            Negative values are clamped to 0.
            
        Examples
        --------
        >>> metabolome.update({'glucose': 15.0, 'lactate': 8.0})
        # Sets glucose to 15 mM, lactate to 8 mM
        """
        # Validate input
        if not isinstance(concentration_dict, dict):
            raise TypeError("concentration_dict must be a dictionary")
        
        # Check that all metabolite names exist
        unknown_metabolites = set(concentration_dict.keys()) - set(self.metabolites)
        if unknown_metabolites:
            raise ValueError(f"Unknown metabolites: {unknown_metabolites}")
        
        # Update concentrations
        for name, value in concentration_dict.items():
            metabolite = self._metabolite_dict[name]
            metabolite.update(value)

    # ----------------------------------------------------------------------- #
    # JSON serialization
    # ----------------------------------------------------------------------- #
    def to_json(self, filename: str = None, full_model: bool = False) -> str:
        """
        Convert the metabolome to JSON format.
        
        Parameters
        ----------
        filename : str, optional
            If provided, save the JSON to this file.
        full_model : bool, default False
            If True, returns a complete model structure ready for visualization.
            If False, returns just the metabolome data.
            
        Returns
        -------
        str
            JSON string representation of the metabolome.
            
        Examples
        --------
        >>> metabolome.to_json()  # Just metabolome data
        >>> model_json = metabolome.to_json(full_model=True)  # Complete model structure
        >>> metabolome.to_json("metabolome.json")  # Save to file
        """
        # Convert metabolites to JSON format
        metabolites_data = {}
        for name in self.metabolites:
            metabolite = self._metabolite_dict[name]
            metabolites_data[name] = {
                "id": metabolite.name,
                "name": metabolite.name,
                "concentration": metabolite.concentration,
                "formula": metabolite.formula,
                "color": metabolite.color
            }
            
            # Only include description if it's not empty
            if metabolite.description:
                metabolites_data[name]["description"] = metabolite.description
        
        metabolome_data = {
            "metabolites": metabolites_data
        }
        
        if full_model:
            # Return complete model structure
            model_data = {
                "metabolome": metabolome_data
            }
            json_str = json.dumps(model_data, indent=2)
        else:
            # Return just metabolome data
            json_str = json.dumps(metabolome_data, indent=2)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(json_str)
        
        return json_str

    # ----------------------------------------------------------------------- #
    # Visualization
    # ----------------------------------------------------------------------- #
    def make_plot(self, title: Optional[str] = None) -> go.Figure:
        """
        Create an interactive Plotly bar plot of metabolite concentrations.
        
        Parameters
        ----------
        title : Optional[str], optional
            Title for the plot. If None, uses "Metabolite Concentrations".
            
        Returns
        -------
        go.Figure
            Interactive Plotly figure with metabolite concentrations.
            
        Notes
        -----
        * Each metabolite is represented as a bar with its assigned color
        * Bars can be toggled on/off by clicking on the legend labels
        * The figure is automatically displayed in Jupyter notebooks
        """
        # Get current concentrations and colors
        concentrations = self.get_concentration()
        colors = [self._metabolite_dict[name].color for name in self.metabolites]
        
        # Create the main bar plot
        fig = go.Figure()
        
        # Add bars for each metabolite
        for i, (name, conc, color) in enumerate(zip(self.metabolites, concentrations, colors)):
            fig.add_trace(go.Bar(
                x=[name],
                y=[conc],
                name=name,
                marker=dict(
                    color=color,
                    line=dict(color='black', width=1.5)
                ),
                hovertemplate=f'<b>{name}</b><br>' +
                             f'Concentration: {conc:.2f} mM<br>' +
                             f'<extra></extra>',
                showlegend=True
            ))
        
        # Update layout
        plot_title = title or "Metabolite Concentrations"
            
        # Optimize layout for large datasets
        legend_config = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
        
        # For large datasets, use vertical legend to save space
        if len(self.metabolites) > 10:
            legend_config.update({
                "orientation": "v",
                "yanchor": "top",
                "y": 1,
                "xanchor": "left",
                "x": 1.02
            })
            
        fig.update_layout(
            title=plot_title,
            xaxis_title="Metabolites",
            yaxis_title="Concentration (mM)",
            template="plotly_white",
            hovermode="closest",
            legend=legend_config,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            tickangle=-45  # Rotate labels for better readability
        )
        fig.update_yaxes(
            zeroline=True,
            zerolinecolor="lightgray",
            gridcolor="lightgray",
            gridwidth=0.5
        )
        
        return fig

    # ----------------------------------------------------------------------- #
    # Convenience methods
    # ----------------------------------------------------------------------- #
    def get_metabolite(self, name: str) -> Metabolite:
        """
        Get a metabolite by name.
        
        Parameters
        ----------
        name : str
            Name of the metabolite.
            
        Returns
        -------
        Metabolite
            The metabolite object.
            
        Raises
        ------
        KeyError
            If the metabolite name is not found.
        """
        if name not in self._metabolite_dict:
            raise KeyError(f"Metabolite '{name}' not found")
        return self._metabolite_dict[name]

    def add_metabolite(self, metabolite: Metabolite) -> None:
        """
        Add a new metabolite to the metabolome.
        
        Parameters
        ----------
        metabolite : Metabolite
            The metabolite to add.
            
        Raises
        ------
        ValueError
            If a metabolite with the same name already exists.
        """
        if metabolite.name in self._metabolite_dict:
            raise ValueError(f"Metabolite '{metabolite.name}' already exists")
        
        self._metabolite_dict[metabolite.name] = metabolite
        self.metabolites = sorted(self._metabolite_dict.keys())
        self.nmets = len(self.metabolites)

    def remove_metabolite(self, name: str) -> None:
        """
        Remove a metabolite from the metabolome.
        
        Parameters
        ----------
        name : str
            Name of the metabolite to remove.
            
        Raises
        ------
        KeyError
            If the metabolite name is not found.
        """
        if name not in self._metabolite_dict:
            raise KeyError(f"Metabolite '{name}' not found")
        
        del self._metabolite_dict[name]
        self.metabolites = sorted(self._metabolite_dict.keys())
        self.nmets = len(self.metabolites)

    # ----------------------------------------------------------------------- #
    # Introspection helpers
    # ----------------------------------------------------------------------- #
    def __repr__(self) -> str:
        return (f"Metabolome(n_metabolites={self.nmets}, "
                f"metabolites={self.metabolites})")

    def __str__(self) -> str:
        concentrations = self.get_concentration()
        metabolite_strs = [f"{name}: {conc:.2f} mM" 
                          for name, conc in zip(self.metabolites, concentrations)]
        return f"Metabolome({', '.join(metabolite_strs)})"

    def __len__(self) -> int:
        return self.nmets

    def __contains__(self, name: str) -> bool:
        return name in self._metabolite_dict

# --------------------------------------------------------------------------- #
# End of file
# --------------------------------------------------------------------------- #
