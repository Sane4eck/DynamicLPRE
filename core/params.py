# core/params.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    rhoFu: float = 814.41
    C1: float = 1.27465e-9

    a01: float = 7.15292e8
    a12: float = 5.8531e12
    a13: float = 1.96184e12

    j01: float = 10115.2
    j12: float = 32228.9
    j13: float = 37852.4

    p2: float = 1e5
    p3: float = 1e5
