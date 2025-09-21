"""
mg_kinetic_model
~~~~~~~~~~~~~~~~

A kinetic modeling engine for microbial community dynamics.

This package provides building blocks such as Metabolite, Metabolome,
and Reactor classes for simulating microbiome dynamics.
"""

__version__ = "0.1.0"

# Public API
from .metabolite import Metabolite
from .metabolome import Metabolome
from .feeding_term import FeedingTerm
from .ph import pH
from .temperature import Temperature
from .stirring import Stirring
from .environment import Environment
from .subpopulation import Subpopulation
from .bacteria import Bacteria, evaluate_transition_condition
from .microbiome import Microbiome
from .reactor import Reactor, Pulse
from .model_from_json import ModelFromJson
from .visualize import GraphSpec, GraphSpecBuilder, GraphSpecConfig, CytoscapeExporter


__all__ = [
    "Metabolite",
    "Metabolome",
    "FeedingTerm",
    "pH",
    "Temperature",
    "Stirring",
    "Environment",
    "Subpopulation",
    "Bacteria",
    "evaluate_transition_condition",
    "Microbiome",
    "Reactor",
    "Pulse",
    "ModelFromJson",
    "GraphSpec",
    "GraphSpecBuilder",
    "GraphSpecConfig",
    "CytoscapeExporter"
]