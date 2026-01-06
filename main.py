# main.py
from core.state import State
from core.params import Params
from core.model import HydraulicModel
from core.integrators import rk4_step
from core.result import Result
import time

dt = 1e-5
endTime = 1.0

state = State()
params = Params()
model = HydraulicModel()
result = Result()

t0 = time.perf_counter()

while state.time < endTime:
    result.save(state)
    rk4_step(model, state, params, dt)

t1 = time.perf_counter()
print(f"Execution time: {t1-t0:.2f} s")

from plot import plot_results
plot_results(result.data)
