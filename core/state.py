from dataclasses import dataclass
import numpy as np


@dataclass
class StateVector:
    names: list[str]
    y0: np.ndarray

    def size(self) -> int:
        return len(self.y0)
