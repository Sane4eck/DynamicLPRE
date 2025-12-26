# models/scheme_valve_system.py

from dataclasses import dataclass
import numpy as np

from models.base_scheme import BaseScheme
from models.signals import ScalarSignal, ConstantSignal

# ---- Індекси змінних стану у векторі y ----
IDX_MAS_B_FU = 0   # маса в паливному балоні
IDX_MAS_B_OX = 1   # маса в окисному балоні
IDX_P_B_OX   = 2   # тиск в окисному балоні
IDX_P1_FU    = 3   # тиск в паливній магістралі 1
IDX_P_GG     = 4   # тиск у газогенераторі
IDX_M_P1_FU  = 5   # mP1Fu
IDX_M_B1_FU  = 6   # mB1Fu
IDX_M1_GG_FU = 7   # m1GgFu
IDX_X_V1_FU  = 8   # хід клапана 1
IDX_U_V1_FU  = 9   # швидкість клапана 1
IDX_X_V2_FU  = 10  # хід клапана 2
IDX_U_V2_FU  = 11  # швидкість клапана 2

N_STATE = 12


@dataclass
class ValveSystemParams:
    """
    Параметри схеми. Значення зараз-заглушки, їх треба буде
    перенести з твоєї WM-програми (або з Excel).
    """

    # Паливний балон
    rho_fu: float = 1000.0        # густина палива, кг/м^3
    p_env: float = 1.0e5          # навколишній тиск, Па
    v_b_fu_nom: float = 0.01      # номінальний об'єм паливного балона, м^3
    p_b_fu_nom: float = 2.0e6     # номінальний тиск у паливному балоні, Па

    # Окисний балон
    rho_ox: float = 1000.0        # густина окислювача
    v_b_ox_nom: float = 0.01      # номінальний об'єм окисного балона, м^3
    p_b_ox_nom: float = 2.0e6     # номінальний тиск в окисному балоні, Па
    T_b_ox0: float = 90.0         # початкова температура в балоні, K
    R_ox: float = 300.0           # газова стала (модельна)
    m_b_ox_const: float = 0.029   # константна витрата з балона (як у WM)
    k_ox_const: float = 1.3       # kBOx (поки константа)

    # Параметри для гідравліки між вузлами (резистивності та інерції)
    resBV1Fu: float = 1.0     # базовий опір ділянки "Ballon Fu -> V1"
    resV11Fu: float = 1.0     # додатковий опір після клапана V1
    resPV2Fu: float = 1.0     # базовий опір ділянки "насос -> V2"
    resV21Fu: float = 1.0     # додатковий опір після клапана V2
    resV3GgFu: float = 1.0    # опір ділянки "V3 -> ГГ"

    coefResisV1Fu: float = 1.0  # коефіцієнт у funcCheckValveCalcResisOtXCV01 для V1
    coefResisV2Fu: float = 1.0  # те саме для V2

    inertB1Fu: float = 1.0    # inertB1Fu
    inertP1Fu: float = 1.0    # inertP1Fu
    inert1GgFu: float = 1.0   # inert1GgFu

    C1: float = 1.0           # ємність вузла 1 (аналог C1 в WM)


class ValveSystemScheme(BaseScheme):
    """
    Поточна реалізація:
    - Ballon Fu (масса + тиск через просту модель)
    - Ballon Ox (масса + температура/об'єм, тиск поки сталий)
    - Гідравлічний ланцюжок Fu: mB1Fu, mP1Fu, m1GgFu, p1Fu
    Клапани поки без власної динаміки (x, u) – вони є в стані, але RHS = 0.
    """

    def __init__(self, params: ValveSystemParams):
        self.params = params
        self.p_pfu_signal: ScalarSignal = ConstantSignal(value=params.p_b_fu_nom)


    def initial_state(self) -> np.ndarray:
        """
        Початковий вектор стану.

        Тут поки стоять «розумні» заготовки:
        - балони «повні» згідно з номінальним об'ємом і густиною;
        - тиски в балонах = номінальні;
        - тиски в магістралі/ГГ = навколишній;
        - масові витрати на всіх ділянках = 0;
        - клапани закриті (хід = 0, швидкість = 0).

        Далі сюди можна напряму перенести InitialData з Wolfram
        або читати ці значення з Excel.
        """
        p = self.params
        y0 = np.zeros(N_STATE, dtype=float)

        # 0: маса в паливному балоні
        # припускаємо, що балон повністю заповнений рідиною
        mas_b_fu0 = p.rho_fu * p.v_b_fu_nom
        y0[IDX_MAS_B_FU] = mas_b_fu0

        # 1: маса в окисному балоні
        mas_b_ox0 = p.rho_ox * p.v_b_ox_nom
        y0[IDX_MAS_B_OX] = mas_b_ox0

        # 2: тиск в окисному балоні
        y0[IDX_P_B_OX] = p.p_b_ox_nom

        # 3: тиск у паливній магістралі 1 – стартуємо з навколишнього
        y0[IDX_P1_FU] = p.p_env

        # 4: тиск у газогенераторі – теж з навколишнього
        y0[IDX_P_GG] = p.p_env

        # 5,6,7: масові витрати/масові змінні – з нуля
        y0[IDX_M_P1_FU] = 0.0
        y0[IDX_M_B1_FU] = 0.0
        y0[IDX_M1_GG_FU] = 0.0

        # 8–11: клапани закриті, без початкової швидкості
        y0[IDX_X_V1_FU] = 0.0
        y0[IDX_U_V1_FU] = 0.0
        y0[IDX_X_V2_FU] = 0.0
        y0[IDX_U_V2_FU] = 0.0

        return y0

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        p = self.params

        # ---- розпаковка стану (аналог {masBFu, masBOx, ...} = y;) ----
        mas_b_fu = y[IDX_MAS_B_FU]
        mas_b_ox = y[IDX_MAS_B_OX]
        p_b_ox   = y[IDX_P_B_OX]
        p1_fu    = y[IDX_P1_FU]
        p_gg     = y[IDX_P_GG]
        m_p1_fu  = y[IDX_M_P1_FU]
        m_b1_fu  = y[IDX_M_B1_FU]
        m1_gg_fu = y[IDX_M1_GG_FU]
        x_v1_fu  = y[IDX_X_V1_FU]
        u_v1_fu  = y[IDX_U_V1_FU]
        x_v2_fu  = y[IDX_X_V2_FU]
        u_v2_fu  = y[IDX_U_V2_FU]

        # ---- ініціалізація похідних (аналог ConstantArray[0, Length[y]]) ----
        dmas_b_fu = 0.0
        dmas_b_ox = 0.0
        dp_b_ox   = 0.0
        dp1_fu    = 0.0
        dp_gg     = 0.0
        dm_p1_fu  = 0.0
        dm_b1_fu  = 0.0
        dm1_gg_fu = 0.0
        dx_v1_fu  = 0.0
        du_v1_fu  = 0.0
        dx_v2_fu  = 0.0
        du_v2_fu  = 0.0

        # ============================================================
        # БЛОК "Ballon Fu"
        #   DmasBFu = -mB1Fu;
        #   vBFu = vBFuNom - Abs[masBFu]/rhoBFu;
        #   pBFu = If[vBFu <= vBFuNom*0.01, pEnv, pBFuNom];
        # ============================================================
        dmas_b_fu = -m_b1_fu

        v_b_fu = p.v_b_fu_nom - abs(mas_b_fu) / p.rho_fu

        if v_b_fu <= p.v_b_fu_nom * 0.01:
            p_b_fu = p.p_env
        else:
            p_b_fu = p.p_b_fu_nom
        # p_b_fu – тиск у паливному балоні (алгебраїчна змінна)

        # ============================================================
        # БЛОК "Ballon Ox"
        #
        #   kBOx = interpKOxygen[pBOx, TBOx];           (поки k_ox_const)
        #   TBOx = TBOx0 * (pBOx/pBOxNom)^((kBOx-1)/kBOx);
        #   vBOx = vBOxNom - Abs[masBOx]/rhoBOx;
        #   DmasBOx = -mBOx;
        #   DpBOx = 0  (* повна формула поки не включена *)
        # ============================================================

        m_b_ox = p.m_b_ox_const
        k_b_ox = p.k_ox_const

        if p.p_b_ox_nom > 0.0 and p_b_ox > 0.0:
            T_b_ox = p.T_b_ox0 * (p_b_ox / p.p_b_ox_nom) ** ((k_b_ox - 1.0) / k_b_ox)
        else:
            T_b_ox = p.T_b_ox0

        v_b_ox = p.v_b_ox_nom - abs(mas_b_ox) / p.rho_ox

        dmas_b_ox = -m_b_ox
        dp_b_ox = 0.0  # як у твоєму поточному WM-коді

        # ============================================================
        # БЛОК "ResisV1 / ResisV2 / ResisV3" + DmB1Fu, DmP1Fu, Dm1GgFu, Dp1Fu
        #
        # В WM:
        #   resV1Fu = funcCheckValveCalcResisOtXCV01[xV1Fu, coefResisV1Fu];
        #   resB1Fu = resBV1Fu + resV1Fu + resV11Fu;
        #   resV2Fu = funcCheckValveCalcResisOtXCV01[xV2Fu, coefResisV2Fu];
        #   resP1Fu = resPV2Fu + resV2Fu + resV21Fu;
        #   resV3Fu = 1/(2*interpFV3Fu[t]^2);
        #   res1GgFu = resP1Fu + resV3Fu + resV3GgFu;
        #
        #   DmB1Fu = F(pBFu, p1Fu, resB1Fu, rhoFu, mB1Fu, inertB1Fu);
        #   DmP1Fu = F(pPFu, p1Fu, resP1Fu, rhoFu, mP1Fu, inertP1Fu);
        #   Dm1GgFu = F(p1Fu, pGg, res1GgFu, rhoFu, m1GgFu, inert1GgFu);
        #   Dp1Fu = (mB1Fu + mP1Fu - m1GgFu)/C1;
        #
        #   де F(...) = (#1-#2-(#3/#4)*Abs[#5]*#5)*(1/#6)
        #
        # Тут funcCheckValveCalcResisOtXCV01 замінена на просту заглушку:
        #   resV ~ coef / x  (будемо уточнювати модель пізніше).
        # ============================================================

        def valve_resis(x: float, coef: float) -> float:
            """Заглушка для funcCheckValveCalcResisOtXCV01."""
            x_eff = float(x)
            if x_eff <= 0.0:
                x_eff = 1e-6
            return coef / x_eff

        # ResisV1 / ResisV2
        resV1Fu = valve_resis(x_v1_fu, p.coefResisV1Fu)
        resB1Fu = p.resBV1Fu + resV1Fu + p.resV11Fu

        resV2Fu = valve_resis(x_v2_fu, p.coefResisV2Fu)
        resP1Fu = p.resPV2Fu + resV2Fu + p.resV21Fu

        # resV3Fu: використовує interpFV3Fu[t]; поки підставимо 1
        resV3Fu = 1.0 / (2.0 * 1.0 ** 2)
        res1GgFu = resP1Fu + resV3Fu + p.resV3GgFu

        rho_fu = p.rho_fu

        # pPFu(t): тиск на вході насоса.
        # Зараз це об'єкт ScalarSignal, який можна легко поміняти
        # на константу / формулу / Excel-інтерполяцію.
        pPFu = self.p_pfu_signal(t)


        # F(p_up, p_down, res, rho, m_dot, inert)
        def calc_dm_dt(p_up, p_down, res, rho, m_dot, inert):
            return (p_up - p_down - (res / rho) * abs(m_dot) * m_dot) * (1.0 / inert)

        dm_b1_fu = calc_dm_dt(p_b_fu, p1_fu, resB1Fu, rho_fu, m_b1_fu, p.inertB1Fu)
        dm_p1_fu = calc_dm_dt(pPFu,  p1_fu, resP1Fu, rho_fu, m_p1_fu, p.inertP1Fu)
        dm1_gg_fu = calc_dm_dt(p1_fu, p_gg,  res1GgFu, rho_fu, m1_gg_fu, p.inert1GgFu)

        # рівняння для тиску p1Fu
        dp1_fu = (m_b1_fu + m_p1_fu - m1_gg_fu) / p.C1

        # ============================================================
        # Інші блоки (газогенератор, камера, динаміка клапанів x/u)
        # додамо окремими кроками.
        # ============================================================

        return np.array([
            dmas_b_fu,
            dmas_b_ox,
            dp_b_ox,
            dp1_fu,
            dp_gg,
            dm_p1_fu,
            dm_b1_fu,
            dm1_gg_fu,
            dx_v1_fu,
            du_v1_fu,
            dx_v2_fu,
            du_v2_fu,
        ], dtype=float)
