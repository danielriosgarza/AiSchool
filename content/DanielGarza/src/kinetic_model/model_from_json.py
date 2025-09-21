# -*- coding: utf-8 -*-
"""
kinetic_model.model_from_json
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build Metabolome, Environment, and Microbiome objects from a JSON export.

This version is intentionally minimal: it preserves your existing behavior and
only changes connections so that condition strings are COMPILED to callables.

Usage
-----
>>> from kinetic_model.model_from_json import ModelFromJson
>>> model = ModelFromJson("tests/assets/bh_bt_ri_complete_model_export.json")
>>> met = model.metabolome
>>> mic = model.microbiome
>>> env = model.environment
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Union

import json

# Imports compatible with both flat scripts and installed package
try:
    from .metabolite import Metabolite
    from .metabolome import Metabolome
    from .subpopulation import Subpopulation
    from .bacteria import Bacteria, evaluate_transition_condition
    from .microbiome import Microbiome
    from .environment import Environment
    from .ph import pH
    from .temperature import Temperature
    from .stirring import Stirring
    from .feeding_term import FeedingTerm
except Exception:  # pragma: no cover
    from metabolite import Metabolite
    from metabolome import Metabolome
    from subpopulation import Subpopulation
    from bacteria import Bacteria, evaluate_transition_condition
    from microbiome import Microbiome
    from environment import Environment
    from ph import pH
    from temperature import Temperature
    from stirring import Stirring
    from feeding_term import FeedingTerm


JsonLike = Union[str, Mapping[str, Any]]


class ModelFromJson:
    """
    Parse a JSON model export into runtime objects.

    Attributes
    ----------
    metabolome : Metabolome
    environment : Environment
    microbiome : Microbiome
    raw : dict
        The raw JSON (as a Python dict).
    """

    def __init__(self, source: JsonLike):
        """
        Parameters
        ----------
        source : str | Mapping
            File path to a JSON export, or a pre-loaded dict with the same shape.
        """
        self.raw: Dict[str, Any] = self._load(source)
        self.metabolome: Optional[Metabolome] = None
        self.environment: Optional[Environment] = None
        self.microbiome: Optional[Microbiome] = None
        self._build_all()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load(self, source: JsonLike) -> Dict[str, Any]:
        if isinstance(source, str):
            with open(source, "r", encoding="utf-8") as f:
                return json.load(f)
        if isinstance(source, Mapping):
            return dict(source)  # shallow copy
        raise TypeError("source must be a file path or a mapping")

    # ------------------------------------------------------------------ #
    # Build pipeline
    # ------------------------------------------------------------------ #
    def _build_all(self) -> None:
        self._build_metabolome()
        self._build_environment()
        self._build_microbiome()

    # ------------------------------------------------------------------ #
    # Metabolome
    # ------------------------------------------------------------------ #
    def _build_metabolome(self) -> None:
        meta = self.raw.get("metabolome", {})
        mets_json = meta.get("metabolites", []) or []
        metabolites: List[Metabolite] = []

        for item in mets_json:
            name = item.get("name") or item.get("id")
            if not name:
                continue
            concentration = float(item.get("concentration", 0.0))
            formula = item.get("formula") or {"C": 0, "H": 0, "O": 0, "N": 0, "S": 0, "P": 0}
            color = item.get("color", "#888888")
            desc = item.get("description", "")
            metabolites.append(
                Metabolite(
                    name=name,
                    concentration=concentration,
                    formula=formula,
                    color=color,
                    description=desc,
                )
            )

        self.metabolome = Metabolome(metabolites=metabolites)

    # ------------------------------------------------------------------ #
    # Environment
    # ------------------------------------------------------------------ #
    def _build_environment(self) -> None:
        assert self.metabolome is not None, "Metabolome must be built first"
        meta = self.raw.get("metabolome", {})
        ph_block = meta.get("pH", {}) or {}

        intercept = float(ph_block.get("baseValue", 7.0))

        # Weights may be provided as:
        #   - dict: {"acetate": -0.05, ...}
        #   - list of {metaboliteName, weight}
        weights: Dict[str, float] = {}
        connected = ph_block.get("connectedMetabolites") or ph_block.get("weights") or []
        if isinstance(connected, dict):
            for k, v in connected.items():
                try:
                    weights[str(k)] = float(v)
                except Exception:
                    pass
        elif isinstance(connected, list):
            for row in connected:
                try:
                    mname = row.get("metaboliteName") or row.get("name") or row.get("id")
                    w = float(row.get("weight", row.get("coeff", 0.0)))
                    if mname:
                        weights[str(mname)] = w
                except Exception:
                    continue

        temperature_c = float(meta.get("temperature", 37.0))
        stirring_rate = float(meta.get("stirring", 1.0))
        stirring_base_std = float(meta.get("stirring_base_std", 0.0))

        ph_obj = pH(self.metabolome, intercept=intercept, met_dictionary=weights)
        temp_obj = Temperature(temperature_c)
        stir_obj = Stirring(rate=stirring_rate, base_std=stirring_base_std)

        self.environment = Environment(ph_obj, stir_obj, temp_obj)

    # ------------------------------------------------------------------ #
    # Microbiome (Bacteria/Subpopulations/Connections)
    # ------------------------------------------------------------------ #
    def _build_microbiome(self) -> None:
        assert self.metabolome is not None, "Metabolome must be built first"

        mic_block = self.raw.get("microbiome", {})
        bacteria_block = mic_block.get("bacteria", {}) or {}

        bacteria_objs: Dict[str, Bacteria] = {}

        for species_key, bdef in bacteria_block.items():
            # ---------------- Subpopulations ----------------
            subpop_cfg: Dict[str, Any] = bdef.get("subpopulations", {}) or {}
            subpop_objs: Dict[str, Subpopulation] = {}

            # Build maps so we can translate between "key" and "name" if needed
            key_to_name: Dict[str, str] = {}
            name_to_key: Dict[str, str] = {}

            for sp_key, sp_row in subpop_cfg.items():
                name = str(sp_row.get("name", sp_key))
                key_to_name[sp_key] = name
                name_to_key[name] = sp_key

                count = float(sp_row.get("count", 0.0))
                species = sp_row.get("species", species_key)
                mumax = float(sp_row.get("mumax", 0.0))
                color = sp_row.get("color", "#aaaaaa")
                state = sp_row.get("state", "active")

                # pH sensitivities
                pHopt = float(sp_row.get("pHopt", 7.0))
                pH_left = float(sp_row.get("pH_sensitivity_left", sp_row.get("pHalpha", 2.0)))
                pH_right = float(sp_row.get("pH_sensitivity_right", sp_row.get("pHalpha", 2.0)))

                # Temperature sensitivities
                Topt = float(sp_row.get("Topt", 37.0))
                t_left = float(sp_row.get("tempSensitivity_left", 5.0))
                t_right = float(sp_row.get("tempSensitivity_right", 2.0))

                # Parse feeding terms from JSON if they exist
                feeding_terms = []
                feeding_terms_json = sp_row.get("feedingTerms", [])
                if isinstance(feeding_terms_json, list):
                    for ft_json in feeding_terms_json:
                        if isinstance(ft_json, dict):
                            ft_id = ft_json.get("id", f"{sp_key}_feeding")
                            met_dict = ft_json.get("metDict", {})
                            if met_dict:
                                try:
                                    # Convert metDict to the format expected by FeedingTerm
                                    # The JSON format is {"metabolite": [yield, monodK]}
                                    # FeedingTerm expects {"metabolite": (yield, monodK)}
                                    # Logic: monodK > 0 = consumption (positive yield), monodK = 0 = production (negative yield)
                                    converted_met_dict = {}
                                    for met_name, values in met_dict.items():
                                        if isinstance(values, (list, tuple)) and len(values) >= 2:
                                            yield_val = float(values[0])
                                            monod_k = float(values[1])
                                            
                                            # Determine if this is consumption or production based on monodK
                                            if monod_k > 0:
                                                # Consumption: keep positive yield, ensure monodK is valid
                                                if monod_k <= 0:
                                                    monod_k = 0.1  # Default monodK for consumed metabolites
                                                converted_met_dict[met_name] = (yield_val, monod_k)
                                            elif monod_k == 0:
                                                # Production: make yield negative
                                                converted_met_dict[met_name] = (-abs(yield_val), 0.0)
                                            else:
                                                # Invalid negative monodK, skip this metabolite
                                                print(f"Warning: Skipping {met_name} with invalid negative monodK: {monod_k}")
                                                continue
                                        elif isinstance(values, (int, float)):
                                            # If only one value provided, assume it's yield with default monodK
                                            converted_met_dict[met_name] = (float(values), 0.1)
                                    
                                    if converted_met_dict:
                                        feeding_terms.append(
                                            FeedingTerm(
                                                id=ft_id,
                                                metDict=converted_met_dict,
                                                metabolome=self.metabolome
                                            )
                                        )
                                except Exception as e:
                                    # If feeding term creation fails, skip it but log the error
                                    print(f"Warning: Could not create feeding term for {sp_key}: {e}")
                                    continue

                subpop_objs[sp_key] = Subpopulation(
                    name=name,
                    count=count,
                    species=species,
                    mumax=mumax,
                    feedingTerms=feeding_terms,
                    pHopt=pHopt,
                    pH_sensitivity_left=pH_left,
                    pH_sensitivity_right=pH_right,
                    Topt=Topt,
                    tempSensitivity_left=t_left,
                    tempSensitivity_right=t_right,
                    state=state,
                    color=color,
                )

            # ---------------- Connections (compile conditions) ----------------
            connections_json = bdef.get("connections", {}) or {}
            connections: Dict[str, List[List[Any]]] = {}

            for source_actual, transitions in connections_json.items():
                # The JSON key might be the display name or the dict key; normalize to key
                source_key = name_to_key.get(source_actual, source_actual)
                compiled_list: List[List[Any]] = []

                for entry in transitions:
                    if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                        # skip unknown/invalid entry
                        continue

                    target_actual, cond_expr, rate = entry
                    target_key = name_to_key.get(target_actual, target_actual)

                    # skip edges to unknown targets
                    if target_key not in subpop_objs:
                        continue

                    cond_str = self._clean_transition_condition(cond_expr)

                    # Empty string (or quoted empty) => unconditional
                    raw = (cond_str or "").strip()
                    raw_unquoted = raw.strip('"').strip("'")
                    if raw_unquoted == "" or "..." in raw:
                        cond_cb = (lambda env, concentrations: 1.0)
                    else:
                        try:
                            cond_cb = evaluate_transition_condition(raw, metabolome=self.metabolome)
                        except Exception:
                            # If compilation fails, fall back to unconditional
                            cond_cb = (lambda env, concentrations: 1.0)

                    # Safe numeric rate
                    try:
                        transition_rate = float(rate)
                    except Exception:
                        transition_rate = 0.0

                    compiled_list.append([target_key, cond_cb, transition_rate])

                connections[source_key] = compiled_list

            # ---------------- Assemble Bacteria ----------------
            species_color = bdef.get("color", "#808080")
            bacteria_objs[species_key] = Bacteria(
                species=bdef.get("species", species_key),
                subpopulations=subpop_objs,
                connections=connections,
                color=species_color,
                metabolome=self.metabolome,  # allow condition callables to access metabolite names
            )

        # ---------------- Assemble Microbiome ----------------
        self.microbiome = Microbiome(
            name=mic_block.get("name", "community"),
            bacteria=bacteria_objs,
            color=mic_block.get("color", "#2ecc71"),
        )

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_transition_condition(expr: Any) -> str:
        """
        Normalize a condition expression from JSON into a plain string.

        - None -> ""
        - Keep as-is for strings
        """
        if expr is None:
            return ""
        if isinstance(expr, str):
            return expr
        # If the JSON ever stores expressions in other shapes, turn to string
        return str(expr)
