from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


class ScalarSignal(ABC):
    @abstractmethod
    def __call__(self, t: float) -> float:
        pass


@dataclass
class ConstantSignal(ScalarSignal):
    value: float
    def __call__(self, t: float) -> float:
        return float(self.value)


@dataclass
class FunctionSignal(ScalarSignal):
    func: Callable[[float], float]
    def __call__(self, t: float) -> float:
        return float(self.func(t))
