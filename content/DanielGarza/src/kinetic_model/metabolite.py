# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
kinetic_model.metabolite
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines :class:`~kinetic_model.metabolite.Metabolite`, one of the core 
building blocks of the kinetic_model package.

A **Metabolite** instance

* stores the current concentration (in mM) – the only mutable field that can
  change during a simulation.
* keeps track of its element counts (currently Carbon, Hydrogen, Oxygen, 
  Nitrogen, Sulfur, and Phosphorus).
* automatically recomputes the total number of element *moles* when
  the concentration is updated.

Installation
------------
First install the package:
    pip install -e .

Then import and use:

>>> from kinetic_model.metabolite import Metabolite
>>> glucose = Metabolite(
...     name="glucose",
...     concentration=10.0,
...     formula={'C': 6, 'H': 12, 'O': 6},
...     color="#ff0000"
... )
>>> glucose.concentration
10.0
>>> glucose.carbons_mol
60.0                      # 10 mM * 6 C
>>> glucose.hydrogens_mol
120.0                     # 10 mM * 12 H
>>> glucose.add(-5.0)      # 5 mM glucose removed
>>> glucose.concentration
5.0
>>> glucose.carbons_mol
30.0
>>> glucose.add(-10.0)     # tries to go negative – clamps to 0
>>> glucose.concentration
0.0
>>> glucose.update(20.0)   # Set concentration to 20 mM
>>> glucose.concentration
20.0
>>> glucose.carbons_mol
120.0                     # 20 mM * 6 C
"""

from __future__ import annotations

import json
import os
from typing import Dict, Union
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_COLOR = "#0093f5"

# Supported chemical elements
SUPPORTED_ELEMENTS = frozenset(['C', 'H', 'O', 'N', 'S', 'P'])


# --------------------------------------------------------------------------- #
# Helper classes
# --------------------------------------------------------------------------- #
@dataclass
class ElementCounts:
    """Immutable container for element counts."""
    carbon: int = 0
    hydrogen: int = 0
    oxygen: int = 0
    nitrogen: int = 0
    sulfur: int = 0
    phosphorus: int = 0
    
    def __post_init__(self):
        """Validate all counts are non-negative integers."""
        for field_name, value in self.__dict__.items():
            if not isinstance(value, int):
                raise TypeError(f"Element count for '{field_name}' must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"Element count for '{field_name}' cannot be negative (got {value})")
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary format."""
        return {
            'C': self.carbon,
            'H': self.hydrogen,
            'O': self.oxygen,
            'N': self.nitrogen,
            'S': self.sulfur,
            'P': self.phosphorus,
        }
    
    @classmethod
    def from_dict(cls, formula: Dict[str, int]) -> ElementCounts:
        """Create ElementCounts from a dictionary formula."""
        # Validate input keys
        invalid_elements = set(formula.keys()) - SUPPORTED_ELEMENTS
        if invalid_elements:
            raise ValueError(f"Unsupported elements: {invalid_elements}. Supported: {sorted(SUPPORTED_ELEMENTS)}")
        
        return cls(
            carbon=formula.get('C', 0),
            hydrogen=formula.get('H', 0),
            oxygen=formula.get('O', 0),
            nitrogen=formula.get('N', 0),
            sulfur=formula.get('S', 0),
            phosphorus=formula.get('P', 0)
        )


# --------------------------------------------------------------------------- #
# Public class
# --------------------------------------------------------------------------- #
class Metabolite:
    """
    Represents a single chemical species that can be added to or removed from a
    reactor.

    Parameters
    ----------
    name : str
        Human‑readable identifier (e.g. ``"glucose"``).
    concentration : float
        Initial concentration in **mM**.  Negative values are clamped to ``0``.
    formula : Dict[str, int]
        Empirical element composition, e.g. ``{'C':6, 'H':12, 'O':6,
        'N':1, 'S':0, 'P':0}``.  Any subset of the six elements is allowed.
    color : str, optional
        Display colour for visualisation.  Defaults to ``"#0093f5"``.
    description : str, optional
        Human-readable description of the metabolite.

    Notes
    -----
    * ``concentration`` is the **single mutable attribute** of a `Metabolite`.  
      Whenever the concentration changes, all internal counters – the
      individual element totals and the derived mole counts – are automatically
      updated via the property setter.
    * The class exposes a convenience property for each of the six most
      common biochemical elements:
      ``carbons_mol``, ``hydrogens_mol``, ``oxygens_mol``,
      ``nitrogens_mol``, ``sulfurs_mol`` and ``phosphorus_mol``.
      These properties return the *total moles* of the element
      (concentration in *mM* × number of atoms per molecule) and are read‑only.
    * ``add()`` adds (or subtracts if negative) to the current concentration.
    * ``update()`` replaces the current concentration with a new value.
    * Both methods respect the zero lower limit (negative results are clamped to 0).
    """
    
    def __init__(
        self,
        name: str,
        concentration: float,
        formula: Dict[str, int],
        color: str = DEFAULT_COLOR,
        description: str = ""
    ) -> None:
        if not isinstance(name, str):
            raise TypeError(f"Name must be a string, got {type(name).__name__}")
        if not name.strip():
            raise ValueError("Name cannot be empty")
        
        self.name = name.strip()
        self.color = color
        self.description = description
        self._element_counts = ElementCounts.from_dict(formula)
        
        # Initialize concentration (this will trigger mole calculations)
        self._concentration = 0.0
        self.concentration = concentration
    
    # -----------------------------------------------------------------------
    # Concentration property (mutable)
    # -----------------------------------------------------------------------
    @property
    def concentration(self) -> float:
        """Current concentration in mM (always ≥ 0)."""
        return self._concentration

    @concentration.setter
    def concentration(self, value: float) -> None:
        """Set a new concentration, clamp to 0, and recompute all mole totals."""
        try:
            val = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Concentration must be numeric, got {value!r}") from exc

        # Handle NaN and negative values
        if val != val or val < 0.0:  # NaN check must come first
            val = 0.0

        self._concentration = val

    # -----------------------------------------------------------------------
    # Concentration mutators
    # -----------------------------------------------------------------------
    def add(self, delta: float) -> None:
        """
        Add (or subtract if negative) to the current concentration.
        
        Parameters
        ----------
        delta : float
            Amount to add (positive) or subtract (negative) from current concentration.
            The result is clamped to 0 if it would go negative.
            
        Examples
        --------
        >>> glucose = Metabolite("glucose", 10.0, {'C': 6, 'H': 12, 'O': 6})
        >>> glucose.add(5.0)    # 10 + 5 = 15 mM
        >>> glucose.concentration
        15.0
        >>> glucose.add(-8.0)   # 15 - 8 = 7 mM
        >>> glucose.concentration
        7.0
        >>> glucose.add(-10.0)  # 7 - 10 = -3, clamped to 0
        >>> glucose.concentration
        0.0
        """
        self.concentration = self.concentration + delta

    def update(self, new_concentration: float) -> None:
        """
        Replace the current concentration with a new value.
        
        Parameters
        ----------
        new_concentration : float
            New concentration value. Negative values are clamped to 0.
            
        Examples
        --------
        >>> glucose = Metabolite("glucose", 10.0, {'C': 6, 'H': 12, 'O': 6})
        >>> glucose.update(15.0)  # Set to 15 mM
        >>> glucose.concentration
        15.0
        >>> glucose.update(-5.0)  # Try to set to -5, clamped to 0
        >>> glucose.concentration
        0.0
        """
        self.concentration = new_concentration

    # -----------------------------------------------------------------------
    # Properties – total moles of each supported element
    # -----------------------------------------------------------------------
    @property
    def carbons_mol(self) -> float:
        """Total moles of carbon (concentration × atom count)."""
        return self._concentration * self._element_counts.carbon
    
    @property
    def hydrogens_mol(self) -> float:
        """Total moles of hydrogen (concentration × atom count)."""
        return self._concentration * self._element_counts.hydrogen
    
    @property
    def oxygens_mol(self) -> float:
        """Total moles of oxygen (concentration × atom count)."""
        return self._concentration * self._element_counts.oxygen
    
    @property
    def nitrogens_mol(self) -> float:
        """Total moles of nitrogen (concentration × atom count)."""
        return self._concentration * self._element_counts.nitrogen
    
    @property
    def sulfurs_mol(self) -> float:
        """Total moles of sulfur (concentration × atom count)."""
        return self._concentration * self._element_counts.sulfur
    
    @property
    def phosphorus_mol(self) -> float:
        """Total moles of phosphorus (concentration × atom count)."""
        return self._concentration * self._element_counts.phosphorus

    # -----------------------------------------------------------------------
    # Helper properties
    # -----------------------------------------------------------------------
    @property
    def formula(self) -> Dict[str, int]:
        """Return a mapping of element → atom count."""
        return self._element_counts.to_dict()
    
    @property
    def element_counts(self) -> Dict[str, int]:
        """Return a mapping of element → atom count (alias for formula)."""
        return self._element_counts.to_dict()

    # -----------------------------------------------------------------------
    # JSON serialization methods
    # -----------------------------------------------------------------------
    def to_json(self, filename: str = None, full_model: bool = False) -> str:
        """
        Convert the metabolite to JSON format.
        
        Parameters
        ----------
        filename : str, optional
            If provided, save the JSON to this file.
        full_model : bool, default False
            If True, returns a complete model structure ready for visualization.
            If False, returns just the metabolite data.
            
        Returns
        -------
        str
            JSON string representation of the metabolite.
            
        Examples
        --------
        >>> glucose = Metabolite("glucose", 10.0, {'C': 6, 'H': 12, 'O': 6})
        >>> json_str = glucose.to_json()  # Just metabolite data
        >>> model_json = glucose.to_json(full_model=True)  # Complete model structure
        >>> glucose.to_json("glucose.json")  # Save to file
        """
        metabolite_data = {
            "id": self.name,
            "name": self.name,
            "concentration": self.concentration,
            "formula": self.formula,
            "color": self.color
        }
        
        # Only include description if it's not empty
        if self.description:
            metabolite_data["description"] = self.description
        
        if full_model:
            # Return complete model structure
            model_data = {
                "metabolome": {
                    "metabolites": {
                        self.name: metabolite_data
                    }
                }
            }
            json_str = json.dumps(model_data, indent=2)
        else:
            # Return just metabolite data
            json_str = json.dumps(metabolite_data, indent=2)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(json_str)
        
        return json_str

    # -----------------------------------------------------------------------
    # Introspection helpers (nice REPL representation)
    # -----------------------------------------------------------------------
    def __repr__(self) -> str:
        counts = self._element_counts
        return (
            f"Metabolite(name={self.name!r}, conc={self.concentration:.4g} mM, "
            f"C={counts.carbon}, H={counts.hydrogen}, O={counts.oxygen}, "
            f"N={counts.nitrogen}, S={counts.sulfur}, P={counts.phosphorus})"
        )

    def __str__(self) -> str:
        return f"{self.name}: {self.concentration:.2f} mM"


# --------------------------------------------------------------------------- #
# End of file
# --------------------------------------------------------------------------- #
