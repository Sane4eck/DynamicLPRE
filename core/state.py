# state.py
from dataclasses import dataclass

@dataclass
class State:
    m01: float = 0.0
    m12: float = 0.0
    m13: float = 0.0
    p1:  float = 1e5
    time: float = 0.0
