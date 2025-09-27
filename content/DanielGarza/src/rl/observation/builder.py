from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from kinetic_model.metabolome import Metabolome
from kinetic_model.microbiome import Microbiome
from .expr import SafeExpr, EvalContext
from ..selectors import resolve_selector


@dataclass
class ObservationSettings:
    include_mets: bool = True
    include_pH_used: bool = True
    include_deltas: bool = True
    include_rates: bool = True
    include_action_echo: bool = True
    species_mode: str = "live"  # off|live|aggregates|summary
    include_species_deltas: bool = False
    include_species_rates: bool = False


class ObservationBuilder:
    def __init__(self, metabolome: Metabolome, microbiome: Microbiome, settings: ObservationSettings):
        self.metabolome = metabolome
        self.microbiome = microbiome
        self.s = settings
        self._species_order = list(microbiome.bacteria.keys())
        self._compiled_features: List[SafeExpr] = []
        self._feature_names: List[str] = []

    def set_feature_expressions(self, exprs: List[str]) -> None:
        self._compiled_features = [SafeExpr(e) for e in exprs]
        self._feature_names = list(exprs)

    def species_live_signals(self, reactor_state: np.ndarray) -> np.ndarray:
        # reactor_state: [V, mets..., subpops...]
        nm = self.metabolome.nmets
        sub_idx = 1 + nm
        live_by_species: Dict[str, float] = {k: 0.0 for k in self._species_order}
        for species_name, bac in self.microbiome.bacteria.items():
            for sp_name, sp in bac.subpopulations.items():
                count = float(reactor_state[sub_idx])
                if getattr(sp, "state", "inactive") == "active":
                    live_by_species[species_name] += count
                sub_idx += 1

        live_counts = np.array([live_by_species[name] for name in self._species_order], dtype=np.float32)
        s = float(np.sum(live_counts))
        if s > 0:
            live_shares = (live_counts / s).astype(np.float32)
        else:
            live_shares = np.zeros_like(live_counts, dtype=np.float32)
        return np.concatenate([live_counts, live_shares]).astype(np.float32)

    def species_live_deltas_and_rates(
        self,
        prev_reactor_state: Optional[np.ndarray],
        reactor_state: np.ndarray,
        dt_hours: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-species live count deltas and rates (counts/hour).

        - When prev_reactor_state is None or dt_hours <= 0, returns zeros.
        - Uses the same species ordering as species_live_signals (builder._species_order).
        - Deltas/rates are computed on live counts (not shares).
        """
        cur = self.species_live_signals(reactor_state)
        n = cur.size // 2
        cur_counts = cur[:n].astype(np.float32)

        if prev_reactor_state is None:
            deltas = np.zeros_like(cur_counts, dtype=np.float32)
            rates = np.zeros_like(cur_counts, dtype=np.float32)
            return deltas, rates

        prev = self.species_live_signals(prev_reactor_state)
        prev_counts = prev[:n].astype(np.float32)

        deltas = (cur_counts - prev_counts).astype(np.float32)
        if dt_hours > 0:
            rates = (deltas / float(dt_hours)).astype(np.float32)
        else:
            rates = np.zeros_like(deltas, dtype=np.float32)
        return deltas, rates

    def build(
        self,
        reactor_state: np.ndarray,
        pH_used: float,
        action_echo: Optional[np.ndarray],
        prev_metabolites: Optional[np.ndarray],
        prev_reactor_state: Optional[np.ndarray],
        dt_hours: float,
        kpi: Optional[Dict[str, float]] = None,
        extra_features: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, List[str]]]:
        parts: List[np.ndarray] = []
        manifest: Dict[str, List[str]] = {"features": []}

        # metabolites (optionally included; still computed for deltas/rates)
        mets = reactor_state[1 : 1 + self.metabolome.nmets].astype(np.float32)
        if self.s.include_mets:
            parts.append(mets)
            manifest["features"].append(f"metabolites[{self.metabolome.nmets}]")

        # pH used
        if self.s.include_pH_used:
            parts.append(np.array([float(pH_used)], dtype=np.float32))
            manifest["features"].append("pH_used")

        # action echo
        if self.s.include_action_echo and action_echo is not None:
            parts.append(action_echo.astype(np.float32))
            manifest["features"].append(f"actuator_echo[{len(action_echo)}]")

        # species signals (live counts + shares)
        if self.s.species_mode == "live":
            live = self.species_live_signals(reactor_state)
            parts.append(live)
            n = live.size // 2
            manifest["features"].append(f"species_live_counts[{n}]")
            manifest["features"].append(f"species_live_shares[{n}]")

            # optional species deltas and rates on live counts
            if self.s.include_species_deltas or self.s.include_species_rates:
                deltas, rates = self.species_live_deltas_and_rates(prev_reactor_state, reactor_state, dt_hours)
                if self.s.include_species_deltas:
                    parts.append(deltas.astype(np.float32))
                    manifest["features"].append(f"species_live_deltas[{deltas.size}]")
                if self.s.include_species_rates:
                    parts.append(rates.astype(np.float32))
                    manifest["features"].append(f"species_live_rates[{rates.size}]")

        # deltas and rates for metabolites
        if self.s.include_deltas:
            if prev_metabolites is None:
                dmet = np.zeros_like(mets, dtype=np.float32)
            else:
                dmet = (mets - prev_metabolites).astype(np.float32)
            parts.append(dmet)
            manifest["features"].append(f"metabolite_deltas[{self.metabolome.nmets}]")

            if self.s.include_rates:
                if dt_hours > 0 and prev_metabolites is not None:
                    rmet = (dmet / float(dt_hours)).astype(np.float32)
                else:
                    rmet = np.zeros_like(dmet, dtype=np.float32)
                parts.append(rmet)
                manifest["features"].append(f"metabolite_rates[{self.metabolome.nmets}]")

        # optional expression-based features (selectors/expressions)
        if self._compiled_features:
            # assemble a namespace for expressions
            feat_ns: Dict[str, float] = {f"met[{i}]": float(m) for i, m in enumerate(mets)}
            feat_ns["pH"] = float(pH_used)
            if action_echo is not None and action_echo.size >= 6:
                feat_ns.update({
                    "action.q": float(action_echo[0]),
                    "action.v": float(action_echo[1]),
                    "action.pH_ctrl": float(action_echo[2]),
                    "action.pH_set": float(action_echo[3]),
                    "action.stir": float(action_echo[4]),
                    "action.temp": float(action_echo[5]),
                })
            # KPI selectors (kpi.*)
            if kpi:
                for kk, vv in kpi.items():
                    key = kk if str(kk).startswith("kpi.") else f"kpi.{kk}"
                    feat_ns[key] = float(vv)
            if extra_features:
                for kk, vv in extra_features.items():
                    try:
                        feat_ns[str(kk)] = float(vv)
                    except (TypeError, ValueError):
                        continue
            # evaluate: permit direct selectors by wrapping as expr if needed
            expr_vals = []
            for expr in self._compiled_features:
                try:
                    expr_vals.append(expr.eval(EvalContext(feat_ns)))
                except Exception:
                    # fallback try selector resolution
                    try:
                        val = resolve_selector(expr.expr, reactor_state, self.metabolome, self.microbiome, float(pH_used), action_echo)
                        expr_vals.append(float(val))
                    except Exception as _:
                        raise
            parts.append(np.array(expr_vals, dtype=np.float32))
            manifest["features"].append(f"expr_features[{len(expr_vals)}]")

        obs = np.concatenate(parts).astype(np.float32)
        return obs, manifest


