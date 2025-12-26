# models/signals.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from data_sources.excel_sources import TimeSeries1D


class ScalarSignal(ABC):
    """Абстрактний скалярний сигнал s(t)."""

    @abstractmethod
    def __call__(self, t: float) -> float:
        raise NotImplementedError


@dataclass
class ConstantSignal(ScalarSignal):
    """s(t) = const."""
    value: float

    def __call__(self, t: float) -> float:
        return float(self.value)


@dataclass
class FuncSignal(ScalarSignal):
    """s(t) = func(t) — довільна Python-функція."""
    func: Callable[[float], float]

    def __call__(self, t: float) -> float:
        return float(self.func(t))


@dataclass
class ExcelSignal(ScalarSignal):
    """s(t) задається таблично з Excel (через TimeSeries1D)."""
    series: TimeSeries1D

    def __call__(self, t: float) -> float:
        return float(self.series(t))
