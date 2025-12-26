# models/scheme_test.py
import numpy as np
from dataclasses import dataclass

from models.base_scheme import BaseScheme

@dataclass
class TestSchemeParams:
    a: float = 1.0   # коефіцієнт у dy/dt = -a*y

class TestScheme(BaseScheme):
    def __init__(self, params: TestSchemeParams):
        self.params = params

    def initial_state(self) -> np.ndarray:
        # y(0) = 1.0, приклад
        return np.array([1.0], dtype=float)

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        a = self.params.a
        dy_dt = -a * y[0]
        return np.array([dy_dt], dtype=float)
