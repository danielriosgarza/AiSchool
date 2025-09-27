from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Callable, Optional

import ast
import operator as op
import numpy as np


AllowedOps = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _safe_num(n):
    if isinstance(n, (int, float)):
        return float(n)
    raise ValueError("non-numeric literal not allowed")


@dataclass
class EvalContext:
    features: Dict[str, float]
    prev_features: Dict[str, float] | None = None
    dt_hours: float = 1.0


class SafeExpr:
    """Safe arithmetic expression evaluator for feature/reward terms.

    Supports numbers, feature names, and a small set of unary/binary ops.
    Provides helpers like abs(), max(), min(), clip().
    """

    def __init__(self, expr: str):
        self.expr = expr
        self._ast = ast.parse(expr, mode="eval")

    def eval(self, ctx: EvalContext) -> float:
        result = self._eval_node(self._ast.body, ctx)
        if isinstance(result, str):
            return result  # Return strings as-is for function arguments
        return float(result)

    def _eval_node(self, node: ast.AST, ctx: EvalContext) -> float:
        # Python >=3.8: constants are ast.Constant
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return node.value  # Return string literals as-is for delta/rate functions
            return _safe_num(node.value)
        # Support attribute access like action.q -> feature key "action.q"
        if isinstance(node, ast.Attribute):
            key = self._attr_to_key(node)
            if key in ctx.features:
                return float(ctx.features[key])
            raise KeyError(f"unknown attribute: {key}")
        # Support subscript like met['butyrate'] -> feature key "met['butyrate']"
        if isinstance(node, ast.Subscript):
            key = self._subscript_to_key(node)
            if key in ctx.features:
                return float(ctx.features[key])
            # If not found, return key string so delta/rate can resolve
            return key
        if isinstance(node, ast.Name):
            if node.id not in ctx.features:
                raise KeyError(f"unknown name: {node.id}")
            return float(ctx.features[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in AllowedOps:
            left = self._eval_node(node.left, ctx)
            right = self._eval_node(node.right, ctx)
            return float(AllowedOps[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in AllowedOps:
            val = self._eval_node(node.operand, ctx)
            return float(AllowedOps[type(node.op)](val))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            # Special handling for delta/rate to allow selector AST as argument
            if fname == "delta":
                if len(node.args) != 1:
                    raise ValueError("delta() takes exactly one argument")
                if ctx.prev_features is None:
                    return 0.0
                key = self._arg_to_selector_key(node.args[0])
                if key is None:
                    return 0.0
                if key not in ctx.features or key not in (ctx.prev_features or {}):
                    return 0.0
                return float(ctx.features[key] - ctx.prev_features[key])
            if fname == "rate":
                if len(node.args) != 1:
                    raise ValueError("rate() takes exactly one argument")
                if ctx.prev_features is None or ctx.dt_hours <= 0:
                    return 0.0
                key = self._arg_to_selector_key(node.args[0])
                if key is None:
                    return 0.0
                if key not in ctx.features or key not in (ctx.prev_features or {}):
                    return 0.0
                delta_val = ctx.features[key] - ctx.prev_features[key]
                return float(delta_val / ctx.dt_hours)

            args = [self._eval_node(a, ctx) for a in node.args]
            if fname == "abs":
                return float(abs(args[0]))
            if fname == "max":
                return float(max(args))
            if fname == "min":
                return float(min(args))
            if fname == "clip":
                x, lo, hi = args
                return float(np.clip(x, lo, hi))
            if fname == "gate":
                if len(args) != 3:
                    raise ValueError("gate(value, threshold, output) takes exactly 3 arguments")
                val, threshold, output = args
                return float(output if val >= threshold else 0.0)
            if fname == "sum":
                if len(args) < 1:
                    raise ValueError("sum() takes at least one argument")
                return float(sum(args))
            if fname == "mean":
                if len(args) < 1:
                    raise ValueError("mean() takes at least one argument")
                return float(sum(args) / len(args))
            if fname == "min":
                if len(args) < 1:
                    raise ValueError("min() takes at least one argument")
                return float(min(args))
            if fname == "max":
                if len(args) < 1:
                    raise ValueError("max() takes at least one argument")
                return float(max(args))
        raise ValueError("disallowed expression construct")

    def _attr_to_key(self, node: ast.Attribute) -> str:
        parts = []
        cur: Optional[ast.AST] = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        else:
            raise ValueError("unsupported attribute base")
        return ".".join(reversed(parts))

    def _subscript_to_key(self, node: ast.Subscript) -> str:
        # Support patterns like met['butyrate'] or species.live_count['bh']
        base = None
        if isinstance(node.value, ast.Name):
            base = node.value.id
        elif isinstance(node.value, ast.Attribute):
            base = self._attr_to_key(node.value)
        else:
            raise ValueError("unsupported subscript base")

        # slice may be Constant (str) in 3.9+
        idx_val = None
        if isinstance(node.slice, ast.Constant):
            idx_val = node.slice.value
        else:
            # Support simple Index(Name/Constant) for older ASTs
            try:
                idx_val = getattr(node.slice, 'value', None)
                if isinstance(idx_val, ast.Constant):
                    idx_val = idx_val.value
            except Exception:
                pass
        if not isinstance(idx_val, str):
            raise ValueError("only string subscripts are supported")
        return f"{base}['{idx_val}']"

    def _arg_to_selector_key(self, arg_node: ast.AST) -> Optional[str]:
        # Allow string literal, attribute, subscript, or bare name
        if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str):
            return arg_node.value
        if isinstance(arg_node, ast.Attribute):
            return self._attr_to_key(arg_node)
        if isinstance(arg_node, ast.Subscript):
            return self._subscript_to_key(arg_node)
        if isinstance(arg_node, ast.Name):
            return arg_node.id
        return None


