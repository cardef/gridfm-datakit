"""GridFM Forge: graph-native dataset compilation for power-system ML."""

from .graph import GraphSample, compile_scenario
from .pipeline import ForgeConfig, build_dataset

__all__ = ["ForgeConfig", "GraphSample", "build_dataset", "compile_scenario"]
