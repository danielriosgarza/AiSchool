from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np


@dataclass
class ActionDecodeSettings:
    bounds: Dict[str, Tuple[float, float]]
    pH_ctrl_threshold: float = 0.5
    smoothness_enabled: bool = False
    max_delta: Dict[str, float] = None  # optional per-dimension delta caps


class ActionSchema:
    """Decode normalized actions in [-1,1] to physical ranges and apply optional smoothness clipping."""

    def __init__(self, settings: ActionDecodeSettings):
        self.s = settings
        self._prev: Optional[Dict[str, float]] = None

    @staticmethod
    def _lin(x: float, lo: float, hi: float) -> float:
        return (x + 1.0) * 0.5 * (hi - lo) + lo

    def decode(self, action: np.ndarray, include_pH: bool = True) -> Dict[str, float]:
        # expected order: [q, v, pH_ctrl_prob, pH_set, stir, temp] if include_pH else [q, v, stir, temp]
        b = self.s.bounds
        if include_pH:
            qn, vn, ctrlp, phn, stn, tpn = [float(x) for x in action]
            q = np.clip(self._lin(qn, *b["q"]), *b["q"])  # type: ignore
            v = np.clip(self._lin(vn, *b["v"]), *b["v"])  # type: ignore
            pH_set = np.clip(self._lin(phn, *b["pH_set"]), *b["pH_set"])  # type: ignore
            stir = np.clip(self._lin(stn, *b["stir"]), *b["stir"])  # type: ignore
            temp = np.clip(self._lin(tpn, *b["temp"]), *b["temp"])  # type: ignore
            pH_ctrl = 1 if ctrlp > self.s.pH_ctrl_threshold else 0
            decoded = {"q": float(q), "v": float(v), "pH_ctrl": pH_ctrl, "pH_set": float(pH_set), "stir": float(stir), "temp": float(temp)}
        else:
            qn, vn, stn, tpn = [float(x) for x in action]
            q = np.clip(self._lin(qn, *b["q"]), *b["q"])  # type: ignore
            v = np.clip(self._lin(vn, *b["v"]), *b["v"])  # type: ignore
            stir = np.clip(self._lin(stn, *b["stir"]), *b["stir"])  # type: ignore
            temp = np.clip(self._lin(tpn, *b["temp"]), *b["temp"])  # type: ignore
            decoded = {"q": float(q), "v": float(v), "stir": float(stir), "temp": float(temp), "pH_ctrl": 0, "pH_set": 0.0}

        if self.s.smoothness_enabled and self._prev is not None and self.s.max_delta:
            clipped = {}
            for k, v in decoded.items():
                if k in self.s.max_delta:
                    md = float(self.s.max_delta[k])
                    v = float(np.clip(v, self._prev[k] - md, self._prev[k] + md))
                clipped[k] = v
            decoded = clipped

        self._prev = dict(decoded)
        return decoded

    def reset(self) -> None:
        """Drop any cached action state (used when smoothness clipping is enabled)."""
        self._prev = None


