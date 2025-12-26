# models/components/valve.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np


class MassFlowModel(ABC):
    """
    Базовий інтерфейс для моделі масової витрати m_dot(t, state).

    state – це вектор стану y (np.ndarray) для всієї системи.
    """

    @abstractmethod
    def __call__(self, t: float, state: np.ndarray) -> float:
        raise NotImplementedError


@dataclass
class ConstantMassFlow(MassFlowModel):
    """
    Найпростіший варіант: ṁ = const.
    """
    value: float

    def __call__(self, t: float, state: np.ndarray) -> float:
        return float(self.value)


@dataclass
class TableMassFlow(MassFlowModel):
    """
    Витрата як функція часу ṁ(t), задана таблично (наприклад, з Excel).

    func_t повинна мати інтерфейс func_t(t: float) -> float.
    Зручно використовувати data_sources.excel_sources.TimeSeries1D.
    """
    func_t: Callable[[float], float]

    def __call__(self, t: float, state: np.ndarray) -> float:
        return float(self.func_t(t))


@dataclass
class SimpleOrificeValve(MassFlowModel):
    """
    Спрощена модель витрати через отвір:
        ṁ = sign(Δp) * C_d * A * sqrt(2 * ρ * |Δp|)

    Δp = p_up - p_down береться з вектора стану по індексах.
    Це ПРИКЛАД. Згодом сюди можна перенести повну funcCheckValve з Mathematica.
    """
    rho: float               # густина рідини
    area: float              # ефективна площа
    discharge_coeff: float   # коефіцієнт витрати C_d
    idx_p_up: int            # індекс тиску "до" клапана у векторі стану
    idx_p_down: int          # індекс тиску "після" клапана у векторі стану

    def __call__(self, t: float, state: np.ndarray) -> float:
        p_up = float(state[self.idx_p_up])
        p_down = float(state[self.idx_p_down])
        dp = p_up - p_down
        if dp == 0.0:
            return 0.0

        sign = 1.0 if dp > 0.0 else -1.0
        m_dot = sign * self.discharge_coeff * self.area * np.sqrt(2.0 * self.rho * abs(dp))
        return float(m_dot)
