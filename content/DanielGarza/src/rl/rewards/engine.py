from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional

from ..observation.expr import SafeExpr, EvalContext


class RewardTerm:
    def __init__(self, expr: str, weight: float = 1.0, deadband: float = 0.0, when: str = "always"):
        self.expr = SafeExpr(expr)
        self.weight = float(weight)
        self.deadband = float(deadband)
        self.when = str(when)

    def compute(self, features: Dict[str, float], prev_features: Optional[Dict[str, float]] = None, dt_hours: float = 1.0) -> float:
        # Episode-type gating (optional)
        et = str(features.get("__episode_type", ""))
        if self.when == "short" and et != "short":
            return 0.0
        if self.when == "long" and et != "long":
            return 0.0
        val = self.expr.eval(EvalContext(features, prev_features=prev_features, dt_hours=float(dt_hours)))
        if abs(val) < self.deadband:
            val = 0.0
        return float(self.weight * val)


class RewardEngine:
    def __init__(self, terms: List[Dict[str, Any]], terminal: List[Dict[str, Any]] | None = None):
        self.terms = [RewardTerm(t["expr"], t.get("weight", 1.0), t.get("deadband", 0.0), t.get("when", "always")) for t in terms]
        self.terminal_terms = [RewardTerm(t["expr"], t.get("weight", 1.0), 0.0, t.get("when", "always")) for t in (terminal or [])]

    def compute_step(self, features: Dict[str, float], prev_features: Optional[Dict[str, float]] = None, dt_hours: float = 1.0) -> Tuple[float, Dict[str, float]]:
        breakdown = {}
        total = 0.0
        for i, term in enumerate(self.terms):
            v = term.compute(features, prev_features=prev_features, dt_hours=dt_hours)
            breakdown[f"term_{i}"] = v
            total += v
        return float(total), breakdown

    def compute_terminal(self, features: Dict[str, float], prev_features: Optional[Dict[str, float]] = None, dt_hours: float = 1.0) -> Tuple[float, Dict[str, float]]:
        breakdown = {}
        total = 0.0
        for i, term in enumerate(self.terminal_terms):
            v = term.compute(features, prev_features=prev_features, dt_hours=dt_hours)
            breakdown[f"terminal_{i}"] = v
            total += v
        return float(total), breakdown


