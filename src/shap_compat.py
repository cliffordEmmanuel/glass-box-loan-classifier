"""Compatibility shims between SHAP and XGBoost.

Why this exists
---------------
XGBoost >= 2.0 serialises ``base_score`` as a bracketed scientific-notation
string (e.g. ``"[5E-1]"``), which SHAP's ``XGBTreeModelLoader`` tries to
parse with a bare ``float()`` call and crashes. The upstream fix lives in
`shap#3513 <https://github.com/shap/shap/issues/3513>`_.

This approach uses a tightly-scoped context manager:
``builtins.float`` is only tolerant of the bracketed format during SHAP
explainer construction, and is restored immediately afterwards. The patch is
a no-op once SHAP upstream ships the fix, so removing this module later
should be a one-line change in ``xai_engine.py``.
"""

from __future__ import annotations

import builtins
import re
from collections.abc import Iterator
from contextlib import contextmanager

_BRACKETED_FLOAT_RE = re.compile(r"^\[[-+0-9.eE]+\]$")

# Snapshot the real ``float`` at import time so the context manager can
# delegate to it without triggering infinite recursion when
# ``builtins.float`` is swapped out below.
_REAL_FLOAT = builtins.float


def _safe_float(val):  # type: ignore[no-untyped-def]
    """Parse either a plain numeric string or XGBoost 2.0's ``'[5E-1]'`` format.

    Always delegates to the *real* builtin float, never to whatever is
    currently bound at ``builtins.float`` (which may be this very function
    while the context manager is active).
    """
    if isinstance(val, str) and _BRACKETED_FLOAT_RE.match(val):
        return _REAL_FLOAT(val.strip("[]"))
    return _REAL_FLOAT(val)


@contextmanager
def shap_xgb_compat() -> Iterator[None]:
    """Context manager: swap ``builtins.float`` for a bracketed-tolerant version.

    Scope is minimal — use it only around ``shap.TreeExplainer(model)``
    construction. The original ``float`` is always restored, even if the
    wrapped code raises.
    """
    original = builtins.float
    builtins.float = _safe_float  # type: ignore[assignment]
    try:
        yield
    finally:
        builtins.float = original  # type: ignore[assignment]
