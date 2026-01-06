# core/state.py
from dataclasses import dataclass

@dataclass
class State:
    time: float = 0.0

    m01: float = 0.0
    m12: float = 0.0
    m13: float = 0.0
    p1:  float = 1e5

    # допоміжні (НЕ інтегруються)
    p0: float = 0.0

    # похідні
    dm01: float = 0.0
    dm12: float = 0.0
    dm13: float = 0.0
    dp1:  float = 0.0
