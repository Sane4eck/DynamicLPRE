# main.py
from core.state import State
from core.params import Params
from core.model import HydraulicModel
from core.result import Result
import time

dt = 1e-7
endTime = 1.0

state = State()
params = Params()
model = HydraulicModel()

t0 = time.perf_counter()

t_arr, y_arr, dy_arr, aux_arr = model.simulate(
    state0=state,
    params=params,
    dt=dt,
    endTime=endTime,
    countPoint=1000,
    backend="numba"
)

res = Result(t_arr, y_arr, dy_arr, aux_arr)

t1 = time.perf_counter()
print(f"Execution time: {t1-t0:.2f} s")

from plot import plot_results
plot_results(res.data, save_dir="out/plots", prefix="run1", show=False)

from core.export import save_to_excel
save_to_excel(res, "out/data/run1.xlsx")

