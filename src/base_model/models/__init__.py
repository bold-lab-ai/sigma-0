"""Sequence backbones and the LOB model.

``registry`` is the only place an architecture name becomes a layer factory.
Checkpoint parameter names come from the Flax class names, not from module
paths, so moving these modules here did not touch any saved checkpoint.
"""

from .registry import (
    DEFAULT_ARCHITECTURE,
    BackboneBuild,
    BackboneDefinition,
    architecture_from_args,
    attention_config_for_width,
    available_architectures,
    build_backbone,
    get_definition,
    recurrent_carry_for_width,
    resolve_architecture,
)


__all__ = [
    "DEFAULT_ARCHITECTURE",
    "BackboneBuild",
    "BackboneDefinition",
    "architecture_from_args",
    "attention_config_for_width",
    "available_architectures",
    "build_backbone",
    "get_definition",
    "recurrent_carry_for_width",
    "resolve_architecture",
]
