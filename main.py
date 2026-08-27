# main.py
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--system", default="sys_two_phase")
ap.add_argument("--input", help="CARPET input file; required for sys_carpet")
ap.add_argument("--properties", help="CARPET property tables; required for sys_carpet")
ap.add_argument("--cells", type=int, default=100, help="number of CARPET axial cells")
ap.add_argument("--dt", type=float)
ap.add_argument("--end-time", type=float)
ap.add_argument("--count-point", type=int)
ap.add_argument("--no-plots", action="store_true")
ap.add_argument("--no-excel", action="store_true")
args = ap.parse_args()

os.environ["DYNAMICS_SYSTEM"] = args.system
if args.system == "sys_carpet":
    os.environ["CARPET_CELLS"] = str(args.cells)

from core.state import State
from core.system import Params
from core.model import HydraulicModel
from core.result import Result

factory = getattr(Params, "from_files", None)
if factory is not None:
    if not args.input or not args.properties:
        ap.error("sys_carpet requires --input and --properties")
    params = factory(args.input, args.properties)
else:
    params = Params()

start_time = float(getattr(params, "start", 0.0))
dt = args.dt if args.dt is not None else float(getattr(params, "dt", 1.0e-7))
end_time = args.end_time if args.end_time is not None else float(getattr(params, "end", 1.0))
count_point = args.count_point
if count_point is None:
    count_point = 1000 if args.system == "sys_carpet" else 10000

state = State(time=start_time, y=None)
model = HydraulicModel()

t0 = time.perf_counter()
t_arr, y_arr, dy_arr, aux_arr = model.simulate(
    state, params, dt, end_time, countPoint=count_point, backend="numba"
)
res = Result(t_arr, y_arr, dy_arr, aux_arr)
t1 = time.perf_counter()
print(f"Execution time: {t1 - t0:.2f} s")

run_id = time.strftime("%Y%m%d_%H%M%S")

if not args.no_plots:
    from core.plotting import save_all_plots
    save_all_plots(res, out_dir=f"out/plots/{run_id}", show=False)

if not args.no_excel:
    from core.export_excel import save_to_excel
    save_to_excel(res, f"out/data/{run_id}.xlsx")
