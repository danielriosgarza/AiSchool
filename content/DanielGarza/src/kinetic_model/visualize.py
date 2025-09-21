# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
kinetic_model.visualize
~~~~~~~~~~~~~~~~~~~~~~~~~~

Build graph specifications (nodes/edges/groups) from your JSON model and export
to Cytoscape.js format.

Quickstart
----------
>>> from kinetic_model.visualize import GraphSpecBuilder, CytoscapeExporter
>>> import json
>>> model = json.load(open("bh_bt_ri_complete_model_export.json", "r"))
>>> builder = GraphSpecBuilder()
>>> spec = builder.build_from_json(model).build()   # or keep lazy
>>> cy = CytoscapeExporter().export(spec, layout="nice")
>>> isinstance(cy, dict)
True
"""


from __future__ import annotations

import json
import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Iterator, Callable, Union
from pathlib import Path

# Configure logging for warnings
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_COLOR = "#0093f5"
EPSILON = 1e-12

# --------------------------------------------------------------------------- #
# Configuration Classes
# --------------------------------------------------------------------------- #
@dataclass
class GraphSpecConfig:
    """Configuration for GraphSpecBuilder behavior."""
    
    validation_level: str = "basic"
    lazy_evaluation: bool = True
    log_warnings: bool = True
    edge_direction_rules: str = "default"
    node_types: List[str] = field(default_factory=lambda: ["metabolite", "subpopulation", "feedingTerm", "pH"])
    edge_types: List[str] = field(default_factory=lambda: ["metabolic", "transition", "pH_effect"])
    
    def with_updates(self, **kwargs) -> GraphSpecConfig:
        """Create a new config with updated values."""
        new_config = GraphSpecConfig(**self.__dict__)
        for key, value in kwargs.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
        return new_config
    
    def from_file(self, config_path: Union[str, Path]) -> GraphSpecConfig:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return self.with_updates(**config_data)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return self
    
    def from_environment(self) -> GraphSpecConfig:
        """Load configuration from environment variables."""
        import os
        
        env_mapping = {
            'GRAPHSPEC_VALIDATION_LEVEL': 'validation_level',
            'GRAPHSPEC_LAZY_EVALUATION': 'lazy_evaluation',
            'GRAPHSPEC_LOG_WARNINGS': 'log_warnings',
            'GRAPHSPEC_EDGE_RULES': 'edge_direction_rules',
        }
        
        updates = {}
        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                if config_key == 'lazy_evaluation':
                    updates[config_key] = value.lower() in ('true', '1', 'yes')
                else:
                    updates[config_key] = value
        
        return self.with_updates(**updates)

# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def _slug(*parts: str) -> str:
    """Create a URL-safe identifier from parts."""
    s = ":".join(p for p in parts if p is not None)
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^a-zA-Z0-9_:\-\.]", "", s)
    return s

def _as_number(x: Any, default: Optional[float] = 0.0) -> float:
    """Safely convert value to number with fallback."""
    try:
        if x is None:
            return 0.0 if default is None else default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return 0.0 if default is None else default
        return v
    except Exception:
        return 0.0 if default is None else default

def _push(lst: List[Dict[str, Any]], item: Optional[Dict[str, Any]]):
    """Safely append item to list if not None."""
    if item is not None:
        lst.append(item)

def _first_model_like(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Extract first model-like object from JSON Schema examples."""
    if isinstance(obj, dict) and not ("metabolome" in obj or "microbiome" in obj):
        ex = obj.get("examples")
        if isinstance(ex, list) and ex:
            for cand in ex:
                if isinstance(cand, dict) and ("metabolome" in cand or "microbiome" in cand):
                    return cand
            if isinstance(ex[0], dict):
                return ex[0]
        ex1 = obj.get("example")
        if isinstance(ex1, dict) and ("metabolome" in ex1 or "microbiome" in ex1):
            return ex1
    return obj

def _items_from_dict_or_list(container, key_name_guess: str, prefix: str):
    """Yield (key, obj) for dict OR list containers."""
    if isinstance(container, dict):
        return list(container.items())
    elif isinstance(container, list):
        out = []
        for i, obj in enumerate(container):
            if not isinstance(obj, dict):
                out.append((f"{prefix}_{i+1}", {"name": str(obj)}))
            else:
                key = obj.get("id") or obj.get("name") or obj.get(key_name_guess) or f"{prefix}_{i+1}"
                out.append((key, obj))
        return out
    else:
        return []

# --------------------------------------------------------------------------- #
# Edge Direction Resolution
# --------------------------------------------------------------------------- #
class EdgeDirectionResolver(ABC):
    """Abstract base for edge direction resolution strategies."""
    
    @abstractmethod
    def resolve_direction(self, rate: float, K: float) -> Tuple[str, str]:
        """Resolve edge direction and rule used."""
        pass

class DefaultDirectionResolver(EdgeDirectionResolver):
    """Default edge direction resolution based on K values and rates."""
    
    def resolve_direction(self, rate: float, K: float) -> Tuple[str, str]:
        """
        Direction rule:
          - K > 0  => ('consumes', 'K')        # metabolite -> feedingTerm
          - K == 0 & |rate| > 0 => ('produces','K') # feedingTerm -> metabolite
          - else fallback to sign(rate)
        """
        if K > EPSILON and abs(rate) > EPSILON:
            return "consumes", "K"
        if K <= EPSILON and abs(rate) > EPSILON:
            return "produces", "K"
        if rate > EPSILON:
            return "consumes", "rateSign"
        if rate < -EPSILON:
            return "produces", "rateSign"
        return "connects", "none"

# --------------------------------------------------------------------------- #
# Exporters
# --------------------------------------------------------------------------- #
class GraphSpecExporter(ABC):
    """Abstract base for graph specification exporters."""

    @abstractmethod
    def export(self, graph_spec: GraphSpec) -> Any:
        """Export GraphSpec to specific format."""
        pass

class CytoscapeExporter(GraphSpecExporter):
    """Export GraphSpec to Cytoscape.js format with comprehensive styling and layout."""

    def __init__(self):
        self._palette = {
            "metabolite":   "#0ea5e9",
            "subpopulation":"#f59e0b",
            "feedingTerm":  "#8b5cf6",
            "pH":           "#ef4444",
            "groupBorder":  "#94a3b8",
            "edgeBase":     "#94a3b8",
            "consumes":     "#3b82f6",
            "produces":     "#16a34a",
            "feeds":        "#64748b",
            "connects":     "#6b7280",
            "controls":     "#f97316",
        }

    def export(self, graph_spec: GraphSpec, *, 
               layout: str = "nice",
               show_edge_labels: bool = False,
               hide_roles: Optional[set] = None) -> Dict[str, Any]:
        """Export to enhanced Cytoscape.js format."""
        # Ensure we have a complete spec
        if hasattr(graph_spec, 'build'):
            graph_spec = graph_spec.build()
        
        # Enrich metadata for better labels and tooltips
        enriched_spec = self._enrich_graph_metadata(graph_spec)
        
        # Convert to Cytoscape format
        elements = self._to_cytoscape_elements(enriched_spec)
        
        # Add layout positions if using preset layout
        if layout in ("nice", "preset_rect"):
            positions = self._compute_nice_rect_positions(enriched_spec)
            for el in elements.get("nodes", []):
                nid = el.get("data", {}).get("id")
                if nid in positions:
                    el["position"] = {"x": positions[nid]["x"], "y": positions[nid]["y"]}
        
        return {
            "elements": elements,
            "layout": {
                "name": "preset" if layout in ("nice", "preset_rect") else layout,
                "animate": True,
                "animationDuration": 1000
            },
            "style": self._get_enhanced_stylesheet(show_edge_labels, hide_roles)
        }

    def _enrich_graph_metadata(self, graph_spec: GraphSpec) -> Dict[str, Any]:
        """Enrich graph metadata with hover labels and edge labels."""
        # Create a copy to avoid modifying the original
        gs = {k: ([x.copy() for x in v] if isinstance(v, list) else v) 
              for k, v in graph_spec.to_dict().items()}
        
        nodes = gs.get("nodes", [])
        edges = gs.get("edges", [])
        
        # Quick index
        by_id = {n["id"]: n for n in nodes}
        
        # --- Node hover labels
        for n in nodes:
            t = n.get("type", "node")
            d = n.get("data", {}) or {}
            label = n.get("label", n["id"])
            
            if t == "metabolite":
                conc = d.get("concentration")
                formula = d.get("formula")
                parts = [f"{label} (metabolite)"]
                if conc is not None: 
                    parts.append(f"conc={conc:g}")
                if isinstance(formula, dict):
                    parts.append("formula=" + "".join(f"{k}{v}" for k, v in formula.items() if v))
                d["hoverLabel"] = " | ".join(parts)
            elif t == "feedingTerm":
                ft_type = d.get("ftType") or d.get("type") or "feeding-term"
                d["hoverLabel"] = f"{label} ({ft_type})"
            elif t == "subpopulation":
                mu = d.get("mumax")
                st = d.get("state")
                bits = [f"{label} (subpop)"]
                if mu is not None: 
                    bits.append(f"μmax={mu:g}")
                if st is not None: 
                    bits.append(f"state={st}")
                d["hoverLabel"] = " | ".join(bits)
            elif t == "pH":
                d["hoverLabel"] = "pH"
            
            n["data"] = d
        
        # --- Edge labels
        for e in edges:
            role = e.get("role", "connects")
            d = e.get("data", {}) or {}
            
            if role in ("consumes", "produces"):
                r = d.get("rate")
                K = d.get("K")
                met = d.get("metKey")
                parts = [role]
                if met: 
                    parts.append(str(met))
                if r is not None: 
                    parts.append(f"r={float(r):.3g}")
                if K is not None: 
                    parts.append(f"K={float(K):.3g}")
                d.setdefault("edgeLabel", " | ".join(parts))
            elif role == "controls":
                c = d.get("coeff")
                d.setdefault("edgeLabel", f"pH coeff={float(c):.3g}" if c is not None else "pH")
            elif role == "feeds":
                d.setdefault("edgeLabel", "feeds")
            elif role == "connects":
                d.setdefault("edgeLabel", "transition")
            
            e["data"] = d
        
        return gs

    def _to_cytoscape_elements(self, graph_spec: Dict[str, Any], 
                              group_child_types: Tuple[str, ...] = ("subpopulation", "feedingTerm")) -> Dict[str, Any]:
        """Convert GraphSpec to Cytoscape elements with proper grouping."""
        nodes = []
        edges = []
        
        # 1) Add group (compound) nodes
        for g in graph_spec.get("groups", []):
            nodes.append({"data": {
                "id": g["id"],
                "label": g.get("label", g["id"]),
                "type": "group"
            }})
        
        # 2) Add normal nodes, restricting which types get a parent
        for n in graph_spec.get("nodes", []):
            ntype = n.get("type", "node")
            d = {
                "id": n["id"],
                "label": n.get("label", n["id"]),
                "type": ntype,
            }
            # Only subpopulations & feeding terms are placed inside the bacteria group
            if ntype in group_child_types and n.get("groupId"):
                d["parent"] = n["groupId"]
            
            # Never let arbitrary data keys sneak in a parent/groupId
            data_extra = {
                k: v for k, v in (n.get("data") or {}).items()
                if k not in ("id", "label", "type", "parent", "groupId")
            }
            if "color" in n:
                data_extra["color"] = n["color"]
            if "size" in n:
                data_extra["size"] = n["size"]
            d.update(data_extra)
            nodes.append({"data": d})
        
        # 3) Edges
        for e in graph_spec.get("edges", []):
            d = {
                "id": e["id"],
                "source": e["source"],
                "target": e["target"],
                "role": e.get("role", "connects"),
            }
            d.update(e.get("data", {}))
            edges.append({"data": d})
        
        return {"nodes": nodes, "edges": edges}

    def _compute_nice_rect_positions(self, graph_spec: Dict[str, Any],
                                   group_child_types: Tuple[str, ...] = ("subpopulation", "feedingTerm"),
                                   inside_row_gap: float = 140.0,
                                   spacing_x: float = 160.0,
                                   ft_band_width: float = 0.55,
                                   met_band_width: float = 0.40,
                                   outside_gap: float = 180.0) -> Dict[str, Dict[str, float]]:
        """Compute tidy rectangular layout positions."""
        nodes = graph_spec.get("nodes", [])
        edges = graph_spec.get("edges", [])
        groups = graph_spec.get("groups", [])
        
        by_id = {n["id"]: n for n in nodes}
        ntype = {n["id"]: n.get("type", "node") for n in nodes}
        group_of = {n["id"]: n.get("groupId") for n in nodes}
        
        # --- Find subpopulations and feeding terms per group
        subs_by_group = {g["id"]: [] for g in groups}
        fts_by_group = {g["id"]: [] for g in groups}
        for n in nodes:
            gid = n.get("groupId")
            if gid and gid in subs_by_group and n.get("type") == "subpopulation":
                subs_by_group[gid].append(n["id"])
            if gid and gid in fts_by_group and n.get("type") == "feedingTerm":
                fts_by_group[gid].append(n["id"])
        
        # --- Map feeding term -> owning subpopulation (from 'feeds' edges)
        ft_owner = {}
        for e in edges:
            if (e.get("role") == "feeds" and 
                ntype.get(e.get("source")) == "subpopulation" and 
                ntype.get(e.get("target")) == "feedingTerm"):
                ft_owner[e["target"]] = e["source"]
        
        # --- Build weight tables: which metabolite is tied to which FT
        cons_by_met = {}  # met -> [(ft, weight)]
        prod_by_met = {}  # met -> [(ft, weight)]
        for e in edges:
            role, s, t = e.get("role"), e.get("source"), e.get("target")
            data = e.get("data", {}) or {}
            w = float(abs(data.get("rateAbs") or data.get("rate") or 1.0))
            if role == "consumes" and ntype.get(s) == "metabolite" and ntype.get(t) == "feedingTerm":
                cons_by_met.setdefault(s, []).append((t, w))
            elif role == "produces" and ntype.get(s) == "feedingTerm" and ntype.get(t) == "metabolite":
                prod_by_met.setdefault(t, []).append((s, w))
        
        def choose_anchor_ft_for_met(m: str) -> Tuple[Optional[str], str]:
            cons = cons_by_met.get(m, [])
            prod = prod_by_met.get(m, [])
            if cons and not prod:
                return max(cons, key=lambda x: x[1])[0], "above"
            if prod and not cons:
                return max(prod, key=lambda x: x[1])[0], "below"
            if cons and prod:
                if sum(w for _, w in cons) >= sum(w for _, w in prod):
                    return max(cons, key=lambda x: x[1])[0], "above"
                else:
                    return max(prod, key=lambda x: x[1])[0], "below"
            return None, "above"
        
        # --- Prepare placement bookkeeping
        pos = {}
        group_rows = {}  # gid -> {'y_ft':..., 'y_sub':..., 'y_top':..., 'y_bot':..., 'xcols':{sub: x}}
        y_cursor = 0.0
        
        # --- Lay out each bacteria group as a tidy block
        for gid in [g["id"] for g in groups]:
            subs = sorted(subs_by_group.get(gid, []), key=lambda i: by_id[i].get("label", i))
            # Assign column x positions by subpopulation
            ncols = max(1, len(subs))
            xcols = {subs[i]: x for i, x in enumerate(self._centered_positions(ncols, spacing_x, 0.0))}
            
            # y levels for this group
            y_ft = y_cursor
            y_sub = y_cursor + inside_row_gap
            y_top = y_ft - outside_gap
            y_bot = y_sub + outside_gap
            
            # place subpops
            for sp in subs:
                pos[sp] = {"x": xcols[sp], "y": y_sub}
            
            # place FTs under each subpop with small horizontal spread inside that column
            fts_per_sub = {sp: [] for sp in subs}
            for ft in fts_by_group.get(gid, []):
                owner = ft_owner.get(ft)
                if owner in fts_per_sub:
                    fts_per_sub[owner].append(ft)
            
            # position FTs
            ft_x = {}
            for sp in subs:
                fts = sorted(fts_per_sub.get(sp, []), key=lambda i: by_id[i].get("label", i))
                if not fts:
                    continue
                local_spacing = spacing_x * ft_band_width
                xs = self._centered_positions(len(fts), local_spacing, xcols[sp])
                for i, ft in enumerate(fts):
                    pos[ft] = {"x": xs[i], "y": y_ft}
                    ft_x[ft] = xs[i]
            
            # remember block metrics
            left = (min(xcols.values()) if xcols else 0.0) - spacing_x * 0.8
            right = (max(xcols.values()) if xcols else 0.0) + spacing_x * 0.8
            group_rows[gid] = {"y_ft": y_ft, "y_sub": y_sub, "y_top": y_top, "y_bot": y_bot,
                               "left": left, "right": right, "xcols": xcols, "ft_x": ft_x}
            
            # vertical separation before next group
            y_cursor = y_bot + outside_gap * 0.8
        
        # --- Place metabolites near their anchor FT's column
        mets = [n["id"] for n in nodes if n.get("type") == "metabolite"]
        bucket = {}  # (gid, row, ft) -> [met ids]
        unanchored_above = []
        
        for m in mets:
            anchor_ft, row = choose_anchor_ft_for_met(m)
            if anchor_ft and anchor_ft in by_id:
                gid = group_of.get(anchor_ft)
                if gid in group_rows:
                    bucket.setdefault((gid, row, anchor_ft), []).append(m)
                    continue
            # no anchor FT -> keep above the first group
            unanchored_above.append(m)
        
        # assign positions inside each bucket, centered around the FT x
        for (gid, row, ft), mlist in bucket.items():
            xs = self._centered_positions(len(mlist), spacing_x * met_band_width, 
                                        group_rows[gid]["ft_x"].get(ft, 0.0))
            y = group_rows[gid]["y_top"] if row == "above" else group_rows[gid]["y_bot"]
            for i, m in enumerate(mlist):
                pos[m] = {"x": xs[i], "y": y}
        
        # any leftover (disconnected) metabolites: spread them above the widest group
        if unanchored_above:
            if group_rows:
                # use the widest block
                gid0 = max(group_rows.keys(), key=lambda g: group_rows[g]["right"] - group_rows[g]["left"])
                L, R = group_rows[gid0]["left"], group_rows[gid0]["right"]
                xs = self._centered_positions(len(unanchored_above), spacing_x * 0.8, (L + R) / 2.0)
                y = group_rows[gid0]["y_top"]
            else:
                xs = self._centered_positions(len(unanchored_above), spacing_x * 0.8, 0.0)
                y = 0.0
            for i, m in enumerate(unanchored_above):
                pos[m] = {"x": xs[i], "y": y}
        
        # --- Place pH to the right of the overall span, vertically centered
        if groups:
            L_all = min(gr["left"] for gr in group_rows.values())
            R_all = max(gr["right"] for gr in group_rows.values())
            top_all = min(gr["y_top"] for gr in group_rows.values())
            bot_all = max(gr["y_bot"] for gr in group_rows.values())
            for n in nodes:
                if n.get("type") == "pH":
                    pos[n["id"]] = {"x": R_all + spacing_x * 1.2, "y": 0.5 * (top_all + bot_all)}
        
        return pos

    def _compute_nice_rect_positions(self, graph_spec: Dict[str, Any],
                                   group_child_types: Tuple[str, ...] = ("subpopulation", "feedingTerm"),
                                   inside_row_gap: float = 140.0,
                                   spacing_x: float = 160.0,
                                   ft_band_width: float = 0.55,
                                   met_band_width: float = 0.40,
                                   outside_gap: float = 180.0) -> Dict[str, Dict[str, float]]:
        """Compute tidy rectangular layout positions."""
        nodes = graph_spec.get("nodes", [])
        edges = graph_spec.get("edges", [])
        groups = graph_spec.get("groups", [])
        
        by_id = {n["id"]: n for n in nodes}
        ntype = {n["id"]: n.get("type", "node") for n in nodes}
        group_of = {n["id"]: n.get("groupId") for n in nodes}
        
        # --- Find subpopulations and feeding terms per group
        subs_by_group = {g["id"]: [] for g in groups}
        fts_by_group = {g["id"]: [] for g in groups}
        for n in nodes:
            gid = n.get("groupId")
            if gid and gid in subs_by_group and n.get("type") == "subpopulation":
                subs_by_group[gid].append(n["id"])
            if gid and gid in fts_by_group and n.get("type") == "feedingTerm":
                fts_by_group[gid].append(n["id"])
        
        # --- Map feeding term -> owning subpopulation (from 'feeds' edges)
        ft_owner = {}
        for e in edges:
            if (e.get("role") == "feeds" and 
                ntype.get(e.get("source")) == "subpopulation" and 
                ntype.get(e.get("target")) == "feedingTerm"):
                ft_owner[e["target"]] = e["source"]
        
        # --- Build weight tables: which metabolite is tied to which FT
        cons_by_met = {}  # met -> [(ft, weight)]
        prod_by_met = {}  # met -> [(ft, weight)]
        for e in edges:
            role, s, t = e.get("role"), e.get("source"), e.get("target")
            data = e.get("data", {}) or {}
            w = float(abs(data.get("rateAbs") or data.get("rate") or 1.0))
            if role == "consumes" and ntype.get(s) == "metabolite" and ntype.get(t) == "feedingTerm":
                cons_by_met.setdefault(s, []).append((t, w))
            elif role == "produces" and ntype.get(s) == "feedingTerm" and ntype.get(t) == "metabolite":
                prod_by_met.setdefault(t, []).append((s, w))
        
        def choose_anchor_ft_for_met(m: str) -> Tuple[Optional[str], str]:
            cons = cons_by_met.get(m, [])
            prod = prod_by_met.get(m, [])
            if cons and not prod:
                return max(cons, key=lambda x: x[1])[0], "above"
            if prod and not cons:
                return max(prod, key=lambda x: x[1])[0], "below"
            if cons and prod:
                if sum(w for _, w in cons) >= sum(w for _, w in prod):
                    return max(cons, key=lambda x: x[1])[0], "above"
                else:
                    return max(prod, key=lambda x: x[1])[0], "below"
            return None, "above"
        
        # --- Prepare placement bookkeeping
        pos = {}
        group_rows = {}  # gid -> {'y_ft':..., 'y_sub':..., 'y_top':..., 'y_bot':..., 'xcols':{sub: x}}
        y_cursor = 0.0
        
        # --- Lay out each bacteria group as a tidy block
        for gid in [g["id"] for g in groups]:
            subs = sorted(subs_by_group.get(gid, []), key=lambda i: by_id[i].get("label", i))
            # Assign column x positions by subpopulation
            ncols = max(1, len(subs))
            xcols = {subs[i]: x for i, x in enumerate(self._centered_positions(ncols, spacing_x, 0.0))}
            
            # y levels for this group
            y_ft = y_cursor
            y_sub = y_cursor + inside_row_gap
            y_top = y_ft - outside_gap
            y_bot = y_sub + outside_gap
            
            # place subpops
            for sp in subs:
                pos[sp] = {"x": xcols[sp], "y": y_sub}
            
            # place FTs under each subpop with small horizontal spread inside that column
            fts_per_sub = {sp: [] for sp in subs}
            for ft in fts_by_group.get(gid, []):
                owner = ft_owner.get(ft)
                if owner in fts_per_sub:
                    fts_per_sub[owner].append(ft)
            
            # position FTs
            ft_x = {}
            for sp in subs:
                fts = sorted(fts_per_sub.get(sp, []), key=lambda i: by_id[i].get("label", i))
                if not fts:
                    continue
                local_spacing = spacing_x * ft_band_width
                xs = self._centered_positions(len(fts), local_spacing, xcols[sp])
                for i, ft in enumerate(fts):
                    pos[ft] = {"x": xs[i], "y": y_ft}
                    ft_x[ft] = xs[i]
            
            # remember block metrics
            left = (min(xcols.values()) if xcols else 0.0) - spacing_x * 0.8
            right = (max(xcols.values()) if xcols else 0.0) + spacing_x * 0.8
            group_rows[gid] = {"y_ft": y_ft, "y_sub": y_sub, "y_top": y_top, "y_bot": y_bot,
                               "left": left, "right": right, "xcols": xcols, "ft_x": ft_x}
            
            # vertical separation before next group
            y_cursor = y_bot + outside_gap * 0.8
        
        # --- Place metabolites near their anchor FT's column
        mets = [n["id"] for n in nodes if n.get("type") == "metabolite"]
        bucket = {}  # (gid, row, ft) -> [met ids]
        unanchored_above = []
        
        for m in mets:
            anchor_ft, row = choose_anchor_ft_for_met(m)
            if anchor_ft and anchor_ft in by_id:
                gid = group_of.get(anchor_ft)
                if gid in group_rows:
                    bucket.setdefault((gid, row, anchor_ft), []).append(m)
                    continue
            # no anchor FT -> keep above the first group
            unanchored_above.append(m)
        
        # assign positions inside each bucket, centered around the FT x
        for (gid, row, ft), mlist in bucket.items():
            xs = self._centered_positions(len(mlist), spacing_x * met_band_width, 
                                        group_rows[gid]["ft_x"].get(ft, 0.0))
            y = group_rows[gid]["y_top"] if row == "above" else group_rows[gid]["y_bot"]
            for i, m in enumerate(mlist):
                pos[m] = {"x": xs[i], "y": y}
        
        # any leftover (disconnected) metabolites: spread them above the widest group
        if unanchored_above:
            if group_rows:
                # use the widest block
                gid0 = max(group_rows.keys(), key=lambda g: group_rows[g]["right"] - group_rows[g]["left"])
                L, R = group_rows[gid0]["left"], group_rows[gid0]["right"]
                xs = self._centered_positions(len(unanchored_above), spacing_x * 0.8, (L + R) / 2.0)
                y = group_rows[gid0]["y_top"]
            else:
                xs = self._centered_positions(len(unanchored_above), spacing_x * 0.8, 0.0)
                y = 0.0
            for i, m in enumerate(unanchored_above):
                pos[m] = {"x": xs[i], "y": y}
        
        # --- Place pH to the right of the overall span, vertically centered
        if groups:
            L_all = min(gr["left"] for gr in group_rows.values())
            R_all = max(gr["right"] for gr in group_rows.values())
            top_all = min(gr["y_top"] for gr in group_rows.values())
            bot_all = max(gr["y_bot"] for gr in group_rows.values())
            for n in nodes:
                if n.get("type") == "pH":
                    pos[n["id"]] = {"x": R_all + spacing_x * 1.2, "y": 0.5 * (top_all + bot_all)}
        
        return pos

    def _centered_positions(self, n: int, spacing: float, x0: float = 0.0) -> List[float]:
        """Compute centered positions for n elements."""
        if n <= 0: 
            return []
        if n == 1: 
            return [x0]
        # symmetric positions around x0 with even spacing
        half = (n - 1) / 2.0
        return [x0 + (i - half) * spacing for i in range(n)]

    def _get_enhanced_stylesheet(self, show_edge_labels: bool = False, 
                                hide_roles: Optional[set] = None) -> List[Dict[str, Any]]:
        """Get enhanced Cytoscape.js styling with rich visual features."""
        if hide_roles is None:
            hide_roles = set()
        
        styles = [
            {"selector": "node", "style": {
                "label": "data(label)",
                "text-wrap": "wrap",
                "text-max-width": 120,
                "font-size": 10,
                "background-color": "data(color)",
                "width": "mapData(size, 0, 1, 30, 80)",
                "height": "mapData(size, 0, 1, 30, 80)",
                "border-width": 1,
                "border-color": "#334155"
            }},
            {"selector": 'node[type = "metabolite"]', "style": {
                "shape": "ellipse", 
                "background-color": "data(color)"
            }},
            {"selector": 'node[type = "subpopulation"]', "style": {
                "shape": "hexagon", 
                "background-color": "data(color)"
            }},
            {"selector": 'node[type = "feedingTerm"]', "style": {
                "shape": "triangle", 
                "background-color": "data(color)"
            }},
            {"selector": 'node[type = "pH"]', "style": {
                "shape": "diamond", 
                "background-color": "data(color)"
            }},
            {"selector": 'node[type = "group"]', "style": {
                "shape": "round-rectangle",
                "background-opacity": 0,
                "z-compound-depth": "bottom",
                "border-width": 6,
                "border-color": self._palette["groupBorder"],
                "padding": "12px"
            }},
            
            {"selector": "edge", "style": {
                "curve-style": "bezier",
                "width": 2,
                "line-color": self._palette["edgeBase"],
                "target-arrow-shape": "vee",
                "target-arrow-color": self._palette["edgeBase"]
            }},
            {"selector": 'edge[role = "feeds"]', "style": {
                "line-color": self._palette["feeds"],
                "target-arrow-color": self._palette["feeds"],
                "target-arrow-shape": "triangle",
                "display": "none" if "feeds" in hide_roles else "element",
            }},
            {"selector": 'edge[role = "consumes"]', "style": {
                "line-color": self._palette["consumes"],
                "target-arrow-color": self._palette["consumes"],
                "target-arrow-shape": "triangle",
                "width": "mapData(rateAbs, 0, 5, 1, 8)",
                "display": "none" if "consumes" in hide_roles else "element",
            }},
            {"selector": 'edge[role = "produces"]', "style": {
                "line-color": self._palette["produces"],
                "target-arrow-color": self._palette["produces"],
                "target-arrow-shape": "triangle",
                "width": "mapData(rateAbs, 0, 5, 1, 8)",
                "display": "none" if "produces" in hide_roles else "element",
            }},
            {"selector": 'edge[role = "connects"]', "style": {
                "line-style": "dashed",
                "line-color": self._palette["connects"],
                "target-arrow-color": self._palette["connects"],
                "display": "none" if "connects" in hide_roles else "element",
            }},
            {"selector": 'edge[role = "controls"]', "style": {
                "line-color": self._palette["controls"],
                "target-arrow-color": self._palette["controls"],
                "target-arrow-shape": "triangle",
                "width": "mapData(coeff, -1, 1, 1, 6)",
                "display": "none" if "controls" in hide_roles else "element",
            }},
            
            # Show richer labels when the element is selected (click)
            {"selector": "node:selected", "style": {
                "label": "data(hoverLabel)",
                "font-size": 11,
                "text-background-color": "#ffffff",
                "text-background-opacity": 0.85,
                "text-background-padding": 2,
                "z-index": 999
            }},
            {"selector": "edge:selected", "style": {
                "label": "data(edgeLabel)",
                "font-size": 10,
                "text-rotation": "autorotate",
                "text-background-color": "#ffffff",
                "text-background-opacity": 0.85,
                "text-background-padding": 2,
                "z-index": 999
            }},
        ]
        
        # Always-on labels if requested
        if show_edge_labels:
            styles.append({"selector": "edge", "style": {
                "label": "data(edgeLabel)",
                "font-size": 9,
                "text-rotation": "autorotate",
                "text-background-color": "#ffffff",
                "text-background-opacity": 0.7,
                "text-background-padding": 2
            }})
        
        return styles

    def _get_enhanced_stylesheet(self, show_edge_labels: bool = False, 
                                hide_roles: Optional[set] = None) -> List[Dict[str, Any]]:
        """Get enhanced Cytoscape.js styling with rich visual features."""
        if hide_roles is None:
            hide_roles = set()
        
        styles = [
            {"selector": "node", "style": {
                "label": "data(label)",
                "text-wrap": "wrap",
                "text-max-width": 120,
                "font-size": 10,
                "background-color": "data(color)",
                "width": "mapData(size, 0, 5, 20, 120)",
                "height": "mapData(size, 0, 5, 20, 120)",
                "border-width": 1,
                "border-color": "#334155"
            }},
            {"selector": 'node[type = "metabolite"]', "style": {
                "shape": "ellipse", 
                "background-color": "data(color)"
            }},
            {"selector": 'node[type = "subpopulation"]', "style": {
                "shape": "hexagon", 
                "background-color": "data(color)"
            }},
            {"selector": 'node[type = "feedingTerm"]', "style": {
                "shape": "triangle", 
                "background-color": "data(color)"
            }},
            {"selector": 'node[type = "pH"]', "style": {
                "shape": "diamond", 
                "background-color": "data(color)"
            }},
            {"selector": 'node[type = "group"]', "style": {
                "shape": "round-rectangle",
                "background-opacity": 0,
                "z-compound-depth": "bottom",
                "border-width": 6,
                "border-color": self._palette["groupBorder"],
                "padding": "12px"
            }},
            
            {"selector": "edge", "style": {
                "curve-style": "bezier",
                "width": 2,
                "line-color": self._palette["edgeBase"],
                "target-arrow-shape": "vee",
                "target-arrow-color": self._palette["edgeBase"]
            }},
            {"selector": 'edge[role = "feeds"]', "style": {
                "line-color": self._palette["feeds"],
                "target-arrow-color": self._palette["feeds"],
                "target-arrow-shape": "triangle",
                "display": "none" if "feeds" in hide_roles else "element",
            }},
            {"selector": 'edge[role = "consumes"]', "style": {
                "line-color": self._palette["consumes"],
                "target-arrow-color": self._palette["consumes"],
                "target-arrow-shape": "triangle",
                "width": "mapData(rateAbs, 0, 5, 1, 8)",
                "display": "none" if "consumes" in hide_roles else "element",
            }},
            {"selector": 'edge[role = "produces"]', "style": {
                "line-color": self._palette["produces"],
                "target-arrow-color": self._palette["produces"],
                "target-arrow-shape": "triangle",
                "width": "mapData(rateAbs, 0, 5, 1, 8)",
                "display": "none" if "produces" in hide_roles else "element",
            }},
            {"selector": 'edge[role = "connects"]', "style": {
                "line-style": "dashed",
                "line-color": self._palette["connects"],
                "target-arrow-color": self._palette["connects"],
                "display": "none" if "connects" in hide_roles else "element",
            }},
            {"selector": 'edge[role = "controls"]', "style": {
                "line-color": self._palette["controls"],
                "target-arrow-color": self._palette["controls"],
                "target-arrow-shape": "triangle",
                "width": "mapData(coeff, -1, 1, 1, 6)",
                "display": "none" if "controls" in hide_roles else "element",
            }},
            
            # Show richer labels when the element is selected (click)
            {"selector": "node:selected", "style": {
                "label": "data(hoverLabel)",
                "font-size": 11,
                "text-background-color": "#ffffff",
                "text-background-opacity": 0.85,
                "text-background-padding": 2,
                "z-index": 999
            }},
            {"selector": "edge:selected", "style": {
                "label": "data(edgeLabel)",
                "font-size": 10,
                "text-rotation": "autorotate",
                "text-background-color": "#ffffff",
                "text-background-opacity": 0.85,
                "text-background-padding": 2,
                "z-index": 999
            }},
        ]
        
        # Always-on labels if requested
        if show_edge_labels:
            styles.append({"selector": "edge", "style": {
                "label": "data(edgeLabel)",
                "font-size": 9,
                "text-rotation": "autorotate",
                "text-background-color": "#ffffff",
                "text-background-opacity": 0.7,
                "text-background-padding": 2
            }})
        
        return styles

    def _get_default_style(self) -> List[Dict[str, Any]]:
        """Get default Cytoscape.js styling (legacy method)."""
        return self._get_enhanced_stylesheet()

# --------------------------------------------------------------------------- #
# Graph Specification Data Classes
# --------------------------------------------------------------------------- #
@dataclass
class GraphSpec:
    """Immutable graph specification for visualization."""
    
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    groups: List[Dict[str, Any]]
    views: List[Dict[str, Any]]
    meta: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "groups": self.groups,
            "views": self.views,
            "meta": self.meta
        }
    
    def to_json(self, filename: Optional[str] = None, indent: int = 2) -> str:
        """Convert to JSON format."""
        json_str = json.dumps(self.to_dict(), indent=indent)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    @classmethod
    def from_json(cls, json_str: str) -> GraphSpec:
        """Create GraphSpec from JSON string."""
        data = json.loads(json_str)
        return cls(**data)
    
    @classmethod
    def from_file(cls, filename: str) -> GraphSpec:
        """Create GraphSpec from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls(**data)

# --------------------------------------------------------------------------- #
# Lazy Graph Specification
# --------------------------------------------------------------------------- #
class LazyGraphSpec:
    """Lazy evaluation wrapper for GraphSpec."""
    
    def __init__(self, builder: GraphSpecBuilder):
        self.builder = builder
        self._spec: Optional[GraphSpec] = None
    
    def __getattr__(self, name: str):
        """Delegate attribute access to the built spec."""
        if self._spec is None:
            self._spec = self.builder.build()
        return getattr(self._spec, name)
    
    def build(self) -> GraphSpec:
        """Force building of the complete spec."""
        if self._spec is None:
            self._spec = self.builder.build()
        return self._spec
    
    def get_nodes(self) -> Iterator[Dict[str, Any]]:
        """Lazy iterator over nodes."""
        return self.builder._build_nodes()
    
    def get_edges(self) -> Iterator[Dict[str, Any]]:
        """Lazy iterator over edges."""
        return self.builder._build_edges()
    
    def get_groups(self) -> Iterator[Dict[str, Any]]:
        """Lazy iterator over groups."""
        return self.builder._build_groups()

# --------------------------------------------------------------------------- #
# Main Builder Class
# --------------------------------------------------------------------------- #
class GraphSpecBuilder:
    """Main class for building graph specifications from JSON schemas."""
    
    def __init__(self, config: Optional[GraphSpecConfig] = None):
        self.config = config or GraphSpecConfig()
        self.direction_resolver = DefaultDirectionResolver()
        self._model: Optional[Dict[str, Any]] = None
        self._meta_id_map: Dict[str, str] = {}
        self._subpop_id_map: Dict[Tuple[str, str], str] = {}
        
        # Configure logging
        if self.config.log_warnings:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.ERROR)
    
    def build_from_json(self, model: Dict[str, Any]) -> Union[GraphSpec, LazyGraphSpec]:
        """Build graph specification from JSON model."""
        self._model = _first_model_like(model) or {}
        
        if self.config.lazy_evaluation:
            return LazyGraphSpec(self)
        else:
            return self.build()
    
    def build(self) -> GraphSpec:
        """Build the complete graph specification."""
        if self._model is None:
            raise ValueError("No model provided. Call build_from_json() first.")
        
        # Initialize containers
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        groups: List[Dict[str, Any]] = []
        views: List[Dict[str, Any]] = []
        
        # Build components
        self._build_metabolites(nodes)
        self._build_ph(nodes, edges)
        self._build_microbiome(nodes, edges, groups)
        
        # Normalize edge magnitudes
        self._normalize_edges(edges)
        
        # Create views
        views = self._create_views()
        
        # Build metadata
        meta = self._build_metadata(edges)
        
        return GraphSpec(nodes=nodes, edges=edges, groups=groups, views=views, meta=meta)
    
    def _scale_node_size(self, concentration: float) -> float:
        """
        Scale metabolite concentration to node size using logarithmic scaling.
        
        This provides better visual distinction across a wide range of concentrations.
        Small concentrations (0-1) get reasonable sizes, while large concentrations
        (10-100+) still show meaningful differences.
        
        Parameters
        ----------
        concentration : float
            Raw concentration value
            
        Returns
        -------
        float
            Scaled size value suitable for cytoscape visualization
        """
        if concentration <= 0:
            return 0.0
        
        # For very small concentrations, use a minimum size
        if concentration < 0.01:
            return 0.3
        
        # Use a combination of linear and logarithmic scaling
        # For concentrations 0.01 to 1: linear scaling from 0.3 to 0.8
        if concentration <= 1.0:
            return 0.3 + 0.5 * concentration
        
        # For concentrations 1 to 10: logarithmic scaling from 0.8 to 1.2
        if concentration <= 10.0:
            return 0.8 + 0.4 * math.log10(concentration)
        
        # For concentrations > 10: additional square root scaling
        # This prevents extremely high concentrations from dominating
        base_size = 0.8 + 0.4 * math.log10(10.0)  # Should be 1.2
        return base_size * math.sqrt(concentration / 10.0)

    def _build_metabolites(self, nodes: List[Dict[str, Any]]) -> None:
        """Build metabolite nodes."""
        metabolome = self._model.get("metabolome") or {}
        metabolites = metabolome.get("metabolites") or {}
        
        for raw_id, m in _items_from_dict_or_list(metabolites, "id", "met"):
            if not isinstance(m, dict):
                m = {"name": str(m)}
            
            met_id = _slug("met", str(raw_id))
            self._meta_id_map[str(raw_id)] = met_id
            if isinstance(m.get("id"), str):
                self._meta_id_map[m["id"]] = met_id
            
            label = m.get("label") or m.get("name") or str(raw_id)
            color = m.get("color")
            # Use explicit size if provided, otherwise fall back to concentration for automatic sizing
            raw_size = _as_number(m.get("size"), _as_number(m.get("concentration"), 0.0))
            # Apply logarithmic scaling for better visual distinction across concentration ranges
            size = self._scale_node_size(raw_size)
            data = {k: v for k, v in m.items() if k not in ("label", "name", "color", "size")}
            data.update({"size": size})
            
            _push(nodes, {
                "id": met_id, "label": label, "type": "metabolite", 
                "color": color, "data": data
            })
    
    def _build_ph(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        """Build pH nodes and edges."""
        metabolome = self._model.get("metabolome") or {}
        ph = metabolome.get("pH") or {}
        
        if isinstance(ph, dict) and ph:
            ph_id = _slug("pH")
            _push(nodes, {
                "id": ph_id, "label": ph.get("label", "pH"), "type": "pH",
                "color": ph.get("color"),
                "data": {k: v for k, v in ph.items() if k != "connectedMetabolites"}
            })
            
            cm = ph.get("connectedMetabolites") or {}
            if isinstance(cm, dict):
                for met_key, coeff in cm.items():
                    met_target = self._meta_id_map.get(str(met_key)) or _slug("met", str(met_key))
                    data = {"kind": "pH_effect", "coeff": _as_number(coeff, 0.0)}
                    data["edgeLabel"] = f"pH coeff={data['coeff']:.3g}"
                    _push(edges, {
                        "id": _slug("edge", "pH", met_target),
                        "source": ph_id, "target": met_target, "role": "controls",
                        "data": data
                    })
            elif isinstance(cm, list):
                for met_key in cm:
                    met_target = self._meta_id_map.get(str(met_key)) or _slug("met", str(met_key))
                    _push(edges, {
                        "id": _slug("edge", "pH", met_target),
                        "source": ph_id, "target": met_target, "role": "controls",
                        "data": {"kind": "pH_effect", "coeff": 0.0, "edgeLabel": "pH"}
                    })
    
    def _build_microbiome(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], groups: List[Dict[str, Any]]) -> None:
        """Build microbiome components (bacteria, subpopulations, feeding terms)."""
        microbiome = self._model.get("microbiome") or {}
        bacteria = microbiome.get("bacteria") or []
        
        for bi, (bname, bac) in enumerate(_items_from_dict_or_list(bacteria, "id", "Bacteria")):
            if not isinstance(bac, dict):
                bac = {"name": bname}
            
            b_label = bac.get("label") or bac.get("name") or bname or f"Bacteria_{bi+1}"
            b_id = _slug("bac", bac.get("id") or b_label)
            
            _push(groups, {
                "id": b_id, "label": b_label, "type": "group", "parentId": None
            })
            
            # Build subpopulations
            self._build_subpopulations(bac, b_id, nodes, edges)
            
            # Build bacteria-level transitions
            self._build_bacteria_transitions(bac, b_id, edges)
    
    def _build_subpopulations(self, bac: Dict[str, Any], b_id: str, 
                            nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        """Build subpopulation nodes and feeding terms."""
        subpops = bac.get("subpopulations") or []
        
        for si, (spname, sp) in enumerate(_items_from_dict_or_list(subpops, "id", "Subpop")):
            if not isinstance(sp, dict):
                sp = {"name": spname}
            
            sp_label = sp.get("label") or sp.get("name") or spname or f"Subpop_{si+1}"
            sp_key = sp.get("id") or sp_label or spname
            sp_id = _slug("sub", b_id, sp_key)
            self._subpop_id_map[(b_id, str(sp_key))] = sp_id
            
            color = sp.get("color")
            # Ensure subpopulations have a minimum visible size for visualization
            raw_size = _as_number(sp.get("size"), 0.0)
            count = _as_number(sp.get("count"), 0.0)
            # Use count as size if no explicit size is provided, with a minimum of 0.3 for visibility
            size = max(raw_size, count, 0.3) if raw_size > 0 or count > 0 else 0.3
            state = sp.get("state")
            mumax = _as_number(sp.get("mumax"), 0.0)
            
            node_data = {
                "size": size, "state": state, "mumax": mumax,
                **{k: v for k, v in sp.items()
                   if k not in ("label", "name", "color", "size", "state", "mumax", "feedingTerms", "connections", "transitions")}
            }
            
            _push(nodes, {
                "id": sp_id, "label": sp_label, "type": "subpopulation",
                "groupId": b_id, "color": color, "data": node_data
            })
            
            # Build feeding terms
            self._build_feeding_terms(sp, sp_id, b_id, nodes, edges)
    
    def _build_feeding_terms(self, sp: Dict[str, Any], sp_id: str, b_id: str,
                            nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        """Build feeding term nodes and metabolite edges."""
        feeding_terms = sp.get("feedingTerms") or {}
        ft_items = list(feeding_terms.items()) if isinstance(feeding_terms, dict) else \
                   [(ft.get("id") or ft.get("name") or f"Term_{i+1}", ft) 
                    for i, ft in enumerate(feeding_terms)]
        
        for fi, (ftname, ft) in enumerate(ft_items):
            if not isinstance(ft, dict):
                ft = {"name": ftname}
            
            ft_label = ft.get("label") or ft.get("name") or ft.get("id") or f"Term_{fi+1}"
            ft_key = ft.get("id") or ft_label
            ft_id = _slug("ft", sp_id, ft_key)
            met_dict = ft.get("metDict") or ft.get("metabolites") or {}
            
            ft_data = {k: v for k, v in ft.items() if k != "metDict"}
            if "type" in ft_data:
                ft_data["ftType"] = ft_data.pop("type")
            ft_data.setdefault("size", 0.35)
            
            _push(nodes, {
                "id": ft_id, "label": ft_label, "type": "feedingTerm",
                "groupId": b_id, "data": ft_data
            })
            
            # subpopulation -> feedingTerm
            _push(edges, {
                "id": _slug("edge", sp_id, ft_id, "feeds"),
                "source": sp_id, "target": ft_id, "role": "feeds",
                "data": {"kind": "feeds", "edgeLabel": "feeds"}
            })
            
            # metabolite edges (directed by K rule)
            self._build_metabolite_edges(met_dict, ft_id, fi, edges)
    
    def _build_metabolite_edges(self, met_dict: Dict[str, Any], ft_id: str, 
                               fi: int, edges: List[Dict[str, Any]]) -> None:
        """Build edges between metabolites and feeding terms."""
        if not isinstance(met_dict, dict):
            return
        
        for met_key, payload in met_dict.items():
            if isinstance(payload, (list, tuple)) and len(payload) >= 1:
                rate = _as_number(payload[0], 0.0)
                K = _as_number(payload[1], 0.0) if len(payload) > 1 else 0.0
            elif isinstance(payload, dict):
                rate = _as_number(payload.get("rate"), 0.0)
                K = _as_number(payload.get("K"), 0.0)
            else:
                rate, K = 0.0, 0.0
            
            role, rule_used = self.direction_resolver.resolve_direction(rate, K)
            
            met_id = self._meta_id_map.get(str(met_key)) or _slug("met", str(met_key))
            label_txt = f"{met_key} | r={rate:.3g}, K={K:.3g}" if role in ("consumes", "produces") else f"{met_key}"
            
            datae = {
                "rate": rate, "rateAbs": abs(rate), "K": K,
                "metKey": str(met_key), "ft": ft_id, "dirFrom": rule_used,
                "edgeLabel": label_txt
            }
            
            if role == "consumes":
                _push(edges, {
                    "id": _slug("edge", met_id, ft_id, role, str(fi)),
                    "source": met_id, "target": ft_id, "role": role, "data": datae
                })
            elif role == "produces":
                _push(edges, {
                    "id": _slug("edge", ft_id, met_id, role, str(fi)),
                    "source": ft_id, "target": met_id, "role": role, "data": datae
                })
            else:
                _push(edges, {
                    "id": _slug("edge", ft_id, met_id, "connects", str(fi)),
                    "source": ft_id, "target": met_id, "role": "connects", "data": datae
                })
    
    def _build_bacteria_transitions(self, bac: Dict[str, Any], b_id: str, edges: List[Dict[str, Any]]) -> None:
        """Build bacteria-level transition edges."""
        bac_conns = bac.get("connections") or {}
        if not isinstance(bac_conns, dict):
            return
        
        for src_key, lst in bac_conns.items():
            # Get the source subpopulation ID
            src_id = self._subpop_id_map.get((b_id, str(src_key)))
            if not src_id:
                # If not found in mapping, create a fallback ID
                src_id = _slug("sub", b_id, str(src_key))
                # Log warning if subpopulation not found
                if self.config.log_warnings:
                    logger.warning(f"Source subpopulation '{src_key}' not found in bacteria '{b_id}', using fallback ID: {src_id}")
            
            if not isinstance(lst, list):
                if self.config.log_warnings:
                    logger.warning(f"Connections for '{src_key}' is not a list: {type(lst)}")
                continue
            
            for i, entry in enumerate(lst):
                if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                    # Our bacteria class format: [target, condition, rate]
                    tgt = entry[0]
                    condition = entry[1]  # condition expression/function
                    rate = entry[2]       # transition rate
                elif isinstance(entry, (list, tuple)) and len(entry) >= 1:
                    # Fallback for other formats
                    tgt = entry[0]
                    condition = entry[1] if len(entry) > 1 else None
                    rate = entry[2] if len(entry) > 2 else None
                elif isinstance(entry, dict):
                    # Dictionary format
                    tgt = entry.get("target") or entry.get("to")
                    condition = entry.get("condition") or entry.get("fn") or entry.get("expr")
                    rate = entry.get("rate") or entry.get("param") or entry.get("params")
                else:
                    continue
                
                if not tgt:
                    if self.config.log_warnings:
                        logger.warning(f"Missing target in connection {i} for source '{src_key}'")
                    continue
                
                # Get the target subpopulation ID
                tgt_id = self._subpop_id_map.get((b_id, str(tgt)))
                if not tgt_id:
                    # If not found in mapping, create a fallback ID
                    tgt_id = _slug("sub", b_id, str(tgt))
                    # Log warning if target subpopulation not found
                    if self.config.log_warnings:
                        logger.warning(f"Target subpopulation '{tgt}' not found in bacteria '{b_id}', using fallback ID: {tgt_id}")
                
                # Create a more descriptive edge label
                if condition and rate is not None:
                    if isinstance(condition, str):
                        edge_label = f"Condition: {condition[:50]}{'...' if len(condition) > 50 else ''} | Rate: {rate}"
                    else:
                        edge_label = f"Transition | Rate: {rate}"
                else:
                    edge_label = "Transition"
                
                _push(edges, {
                    "id": _slug("edge", src_id, tgt_id, "transition", str(i)),
                    "source": src_id, "target": tgt_id, "role": "connects",
                    "data": {
                        "kind": "transition", 
                        "condition": condition, 
                        "rate": rate, 
                        "edgeLabel": edge_label,
                        "source_subpop": src_key,
                        "target_subpop": tgt
                    }
                })
    
    def _normalize_edges(self, edges: List[Dict[str, Any]]) -> None:
        """Normalize edge magnitudes for consistent styling."""
        max_rate = max((e.get("data", {}).get("rateAbs", 0.0) for e in edges), default=0.0)
        max_coeff = max((abs(e.get("data", {}).get("coeff", 0.0)) for e in edges if e.get("role") == "controls"), default=0.0)
        
        for e in edges:
            d = e.get("data", {})
            if "rateAbs" in d and max_rate > 0:
                d["rateNorm"] = d["rateAbs"] / max_rate
            if e.get("role") == "controls":
                c = abs(d.get("coeff", 0.0))
                d["coeffNorm"] = (c / max_coeff) if max_coeff > 0 else 0.0
    
    def _create_views(self) -> List[Dict[str, Any]]:
        """Create default layout views."""
        default_layouts = ("fcose", "dagre", "concentric")
        views = []
        for name in default_layouts:
            views.append({"id": name, "layout": name})
        return views
    
    def _build_metadata(self, edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build metadata for reporting and debugging."""
        direction_counts = {"K": 0, "rateSign": 0, "none": 0}
        
        for e in edges:
            if "dirFrom" in e.get("data", {}):
                rule = e["data"]["dirFrom"]
                direction_counts[rule] = direction_counts.get(rule, 0) + 1
        
        max_rate = max((e.get("data", {}).get("rateAbs", 0.0) for e in edges), default=0.0)
        max_coeff = max((abs(e.get("data", {}).get("coeff", 0.0)) for e in edges if e.get("role") == "controls"), default=0.0)
        
        return {
            "direction_rule": "K>0 ⇒ consumes (met→FT); K==0 & |rate|>0 ⇒ produces (FT→met); else sign(rate)",
            "direction_counts": direction_counts,
            "edge_stats": {"max_rateAbs": max_rate, "max_abs_coeff": max_coeff}
        }
    
    # Lazy evaluation methods
    def _build_nodes(self) -> Iterator[Dict[str, Any]]:
        """Lazy iterator over nodes."""
        # This would be implemented for true lazy evaluation
        # For now, return the built nodes
        spec = self.build()
        yield from spec.nodes
    
    def _build_edges(self) -> Iterator[Dict[str, Any]]:
        """Lazy iterator over edges."""
        spec = self.build()
        yield from spec.edges
    
    def _build_groups(self) -> Iterator[Dict[str, Any]]:
        """Lazy iterator over groups."""
        spec = self.build()
        yield from spec.groups

    def _enrich_graph_metadata(self, graph_spec: GraphSpec) -> Dict[str, Any]:
        """Enrich graph metadata with hover labels and edge labels."""
        # Create a copy to avoid modifying the original
        gs = {k: ([x.copy() for x in v] if isinstance(v, list) else v) 
              for k, v in graph_spec.to_dict().items()}
        
        nodes = gs.get("nodes", [])
        edges = gs.get("edges", [])
        
        # Quick index
        by_id = {n["id"]: n for n in nodes}
        
        # --- Node hover labels
        for n in nodes:
            t = n.get("type", "node")
            d = n.get("data", {}) or {}
            label = n.get("label", n["id"])
            
            if t == "metabolite":
                conc = d.get("concentration")
                formula = d.get("formula")
                parts = [f"{label} (metabolite)"]
                if conc is not None: 
                    parts.append(f"conc={conc:g}")
                if isinstance(formula, dict):
                    parts.append("formula=" + "".join(f"{k}{v}" for k, v in formula.items() if v))
                d["hoverLabel"] = " | ".join(parts)
            elif t == "feedingTerm":
                ft_type = d.get("ftType") or d.get("type") or "feeding-term"
                d["hoverLabel"] = f"{label} ({ft_type})"
            elif t == "subpopulation":
                mu = d.get("mumax")
                st = d.get("state")
                bits = [f"{label} (subpop)"]
                if mu is not None: 
                    bits.append(f"μmax={mu:g}")
                if st is not None: 
                    bits.append(f"state={st}")
                d["hoverLabel"] = " | ".join(bits)
            elif t == "pH":
                d["hoverLabel"] = "pH"
            
            n["data"] = d
        
        # --- Edge labels
        for e in edges:
            role = e.get("role", "connects")
            d = e.get("data", {}) or {}
            
            if role in ("consumes", "produces"):
                r = d.get("rate")
                K = d.get("K")
                met = d.get("metKey")
                parts = [role]
                if met: 
                    parts.append(str(met))
                if r is not None: 
                    parts.append(f"r={float(r):.3g}")
                if K is not None: 
                    parts.append(f"K={float(K):.3g}")
                d.setdefault("edgeLabel", " | ".join(parts))
            elif role == "controls":
                c = d.get("coeff")
                d.setdefault("edgeLabel", f"pH coeff={float(c):.3g}" if c is not None else "pH")
            elif role == "feeds":
                d.setdefault("edgeLabel", "feeds")
            elif role == "connects":
                d.setdefault("edgeLabel", "transition")
            
            e["data"] = d
        
        return gs
