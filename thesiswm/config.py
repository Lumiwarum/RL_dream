"""
Config dataclasses are intentionally light: Hydra+YAML is the primary config mechanism.

This module is mostly for shared constants and type hints.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Horizons:
    fixed: int = 15
    adaptive: List[int] = None
