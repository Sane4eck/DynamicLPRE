import numpy as np
from core.scheme import DynamicScheme


class ValveSystemModel(DynamicScheme):
    """
    Тут ПОВНІСТЮ твоя математика.
    """

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        # Розпаковка стану (1:1 з WM)
        (
            mBFu, mBOx, pBFu, pBOx,
            p1Fu, mB1Fu, mP1Fu, m1GgFu,
            pGg, mTurb, nPump, ...
        ) = y

        # ==== ТУТ ТИ ВСТАВЛЯЄШ RHS З WM ====

        DmBFu = ...
        DmBOx = ...
        Dp1Fu = ...
        DpGg  = ...

        return np.array([
            DmBFu,
            DmBOx,
            0.0,   # якщо pBFu = const
            0.0,
            Dp1Fu,
            ...
        ], dtype=float)
