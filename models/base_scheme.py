# models/base_scheme.py
from abc import ABC, abstractmethod
import numpy as np

class BaseScheme(ABC):
    """Базовий інтерфейс для будь-якої схеми (будь-який набір ОДУ)."""

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        """Початковий вектор стану y0."""
        raise NotImplementedError

    @abstractmethod
    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Праві частини dy/dt."""
        raise NotImplementedError
