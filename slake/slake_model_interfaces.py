"""Load existing project model interfaces without changing their source files."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

BACKEND_SPECS = {
    "base": ("test/inferQWen3vl.py", "BaselineModelInterface"),
    "mmrl": ("test/inferEngine.py", "ModelInterface"),
    "lora": ("loraTest/loraTest.py", "LoraModelInterface"),
    "lora-vision": ("loraTest/loraVisionTest.py", "LoraModelInterface"),
    "lora-vision-last8": (
        "loraTest/loraLast8VisionExperiments.py",
        "LoraModelInterface",
    ),
    "dora": ("loraTest/doraTest.py", "DoraModelInterface"),
    "dora-vision": ("loraTest/doraVisionTest.py", "DoraModelInterface"),
    "ia3": ("loraTest/ia3Test.py", "IA3ModelInterface"),
    "adapter": ("loraTest/adapterTest.py", "AdapterModelInterface"),
}


def _load_source_module(relative_path: str) -> Any:
    source_path = (REPO_ROOT / relative_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Model interface script not found: {source_path}")

    for import_root in (REPO_ROOT, source_path.parent):
        import_root_text = str(import_root)
        if import_root_text not in sys.path:
            sys.path.insert(0, import_root_text)

    module_name = f"slake_dynamic_{source_path.stem}_{abs(hash(source_path))}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model interface from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_slake_model_interface(
    backend: str,
    base_model_path: str,
    checkpoint_path: str | None = None,
) -> Any:
    """Instantiate one existing project model interface for SLAKE evaluation."""

    if backend not in BACKEND_SPECS:
        raise ValueError(
            f"Unsupported backend={backend!r}; choices={sorted(BACKEND_SPECS)}"
        )
    relative_path, class_name = BACKEND_SPECS[backend]
    module = _load_source_module(relative_path)
    interface_class = getattr(module, class_name)

    if backend == "base":
        return interface_class(base_model_path)
    if not checkpoint_path:
        raise ValueError(f"--checkpoint is required for backend={backend}")
    if backend == "mmrl":
        return interface_class(checkpoint_path, base_model_path)
    if backend == "lora-vision-last8":
        return interface_class(Path(checkpoint_path).resolve(), base_model_path)
    return interface_class(checkpoint_path, base_model_path)
