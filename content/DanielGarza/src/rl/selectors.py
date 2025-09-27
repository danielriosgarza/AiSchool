from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from kinetic_model.metabolome import Metabolome
from kinetic_model.microbiome import Microbiome


def _extract_live_by_species(reactor_state: np.ndarray, microbiome: Microbiome, metabolome: Metabolome) -> Dict[str, float]:
    nm = metabolome.nmets
    sub_idx = 1 + nm
    live: Dict[str, float] = {k: 0.0 for k in microbiome.bacteria.keys()}
    for species_name, bac in microbiome.bacteria.items():
        for _, sp in bac.subpopulations.items():
            count = float(reactor_state[sub_idx])
            if getattr(sp, "state", "inactive") == "active":
                live[species_name] += count
            sub_idx += 1
    return live


def resolve_selector(
    selector: str,
    reactor_state: np.ndarray,
    metabolome: Metabolome,
    microbiome: Microbiome,
    pH_used: float,
    action_echo: np.ndarray,
) -> float:
    s = selector.strip()
    # pH.used
    if s == "pH.used" or s == "pH":
        return float(pH_used)
    # action.* mapping (q,v,pH_ctrl,pH_set,stir,temp)
    if s.startswith("action."):
        idx_map = {"q": 0, "v": 1, "pH_ctrl": 2, "pH_set": 3, "stir": 4, "temp": 5}
        key = s.split(".", 1)[1]
        if key in idx_map and action_echo is not None and action_echo.size >= 6:
            return float(action_echo[idx_map[key]])
    # met['name']
    if s.startswith("met["):
        name = s[len("met["):].strip()
        name = name.strip("]").strip().strip("\"").strip("'")
        if name in metabolome._metabolite_dict:
            idx = list(metabolome.metabolites).index(name)
            return float(reactor_state[1 + idx])
    # species.live_count['species'] or species.live_share['species']
    if s.startswith("species.live_"):
        if s.startswith("species.live_count"):
            species = s.split("[", 1)[1].rstrip("]").strip().strip("\"").strip("'")
            live = _extract_live_by_species(reactor_state, microbiome, metabolome)
            return float(live.get(species, 0.0))
        if s.startswith("species.live_share"):
            species = s.split("[", 1)[1].rstrip("]").strip().strip("\"").strip("'")
            live = _extract_live_by_species(reactor_state, microbiome, metabolome)
            total = float(sum(live.values()))
            return float((live.get(species, 0.0) / total) if total > 0 else 0.0)
    raise ValueError(f"Unsupported selector: {selector}")


