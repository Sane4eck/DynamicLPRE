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

    def as_tuple(self):
        # порядок ФІКСОВАНИЙ і використовується всюди
        return (self.rhoFu, self.C1,
                self.a01, self.a12, self.a13,
                self.j01, self.j12, self.j13,
                self.p2, self.p3)
