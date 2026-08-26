"""PathVQA aliases for the project's existing inference backends."""

from slake.slake_model_interfaces import BACKEND_SPECS, load_slake_model_interface


def load_pathvqa_model_interface(
    backend: str,
    base_model_path: str,
    checkpoint_path: str | None = None,
):
    return load_slake_model_interface(
        backend,
        base_model_path=base_model_path,
        checkpoint_path=checkpoint_path,
    )


__all__ = ["BACKEND_SPECS", "load_pathvqa_model_interface"]
