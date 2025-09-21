# -*- coding: utf-8 -*-
"""
kinetic_model.utils
~~~~~~~~~~~~~~~~~~~~~~

Utilities shared across the package.
"""

from __future__ import annotations
from typing import Any, Mapping
import math
import numpy as np

def json_safe(
    obj: Any,
    *,
    callable_policy: str = "placeholder",   # "placeholder" | "repr" | "skip"
    callable_placeholder: str = "<callable_condition>",
    max_depth: int = 20,
) -> Any:
    """Recursively convert objects to JSON-serializable forms.

    - Callables become a placeholder (or repr/None depending on policy).
    - NumPy scalars/arrays become Python numbers/lists.
    - Non-serializable fall back to str(...) after max_depth.
    """
    if max_depth < 0:
        return str(obj)

    # primitives
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj

    # numpy numbers/arrays
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # mappings / sequences
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v, callable_policy=callable_policy,
                                  callable_placeholder=callable_placeholder,
                                  max_depth=max_depth - 1)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v, callable_policy=callable_policy,
                          callable_placeholder=callable_placeholder,
                          max_depth=max_depth - 1)
                for v in obj]

    # callables
    if callable(obj):
        if callable_policy == "repr":
            return repr(obj)
        if callable_policy == "skip":
            return None
        return callable_placeholder

    # last resort
    return str(obj)
