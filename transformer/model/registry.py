"""Architecture registry for pluggable model factories.

Built-in architectures (encoder_decoder, encoder_only, decoder_only) are
registered automatically when you import transformer.model.

To register a custom architecture:

    from transformer.model.registry import register_architecture

    class MyLLM(nn.Module):
        def __init__(self, cfg): ...
        def forward(self, src, src_mask=None): ...
        def count_parameters(self): ...

    register_architecture("my_llm", lambda cfg: MyLLM(cfg))

Then set  model.architecture: my_llm  in your YAML config and run train.py.
"""
from __future__ import annotations

from typing import Callable, Dict, List

_REGISTRY: Dict[str, Callable] = {}


def register_architecture(name: str, factory: Callable) -> None:
    """Register a model factory function under the given name.

    Args:
        name:    Unique string identifier (used as model.architecture in YAML).
        factory: Callable that takes a Config object and returns an nn.Module.
                 The module must expose .count_parameters(), .pad_id, and
                 .output_proj.out_features for the Trainer to work without changes.
    """
    _REGISTRY[name] = factory


def build_model(architecture: str, cfg):
    """Instantiate the model registered under `architecture`.

    Args:
        architecture: Name as registered via register_architecture().
        cfg:          A Config object (from transformer.config.config.load_config).

    Returns:
        An nn.Module instance.

    Raises:
        ValueError if the architecture name is not registered.
    """
    if architecture not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture: {architecture!r}. "
            f"Registered architectures: {sorted(_REGISTRY)}. "
            "To add a custom one: "
            "from transformer.model.registry import register_architecture; "
            "register_architecture('name', factory_fn)"
        )
    return _REGISTRY[architecture](cfg)


def list_architectures() -> List[str]:
    """Return a sorted list of all registered architecture names."""
    return sorted(_REGISTRY.keys())
