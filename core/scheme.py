from abc import ABC, abstractmethod
import numpy as np


class DynamicScheme(ABC):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        pass
