"""RSVQA aliases for the project's existing inference backends.

The source module is loaded directly so a CPU-only data audit does not execute
``slake.__init__`` and import the training stack (notably PyTorch).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[1] / "slake" / "slake_model_interfaces.py"
_SPEC = importlib.util.spec_from_file_location("rsvqa_slake_model_interfaces", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load shared model interfaces from {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
BACKEND_SPECS = _MODULE.BACKEND_SPECS
load_slake_model_interface = _MODULE.load_slake_model_interface


def load_rsvqa_model_interface(
    backend: str,
    base_model_path: str,
    checkpoint_path: str | None = None,
    interface_kwargs=None,
):
    return load_slake_model_interface(
        backend,
        base_model_path=base_model_path,
        checkpoint_path=checkpoint_path,
        interface_kwargs=interface_kwargs,
    )


__all__ = ["BACKEND_SPECS", "load_rsvqa_model_interface"]
