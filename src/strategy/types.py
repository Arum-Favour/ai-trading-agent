from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


@dataclass
class Signal:
    side: Side
    confidence: float  # 0..1
    reason: str
