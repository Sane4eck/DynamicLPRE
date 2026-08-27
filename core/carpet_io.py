from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")


def _lines(path: str | Path) -> list[str]:
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp1251", "latin1"):
        try:
            return raw.decode(encoding).splitlines()
        except UnicodeDecodeError:
            pass
    raise ValueError(f"Не вдалося визначити кодування {path}")


def numbers(line: str) -> list[float]:
    return [float(x.replace("D", "E").replace("d", "e")) for x in NUMBER.findall(line)]


@dataclass(frozen=True)
class InputData:
    start: float
    end: float
    dt: float
    print_every: int
    snapshots: tuple[float, ...]
    heat: int
    direction: int
    test: int
    pressure: float
    inlet_celsius: float
    delta_t: float
    mass_flow: float
    mass_flux: float
    length: float
    diameter: float
    wall_thickness: float
    metal: int
    wall_temperature: float
    environment_pressure: float
    external_htc: float
    ajo_factor: float
    ajt_factor: float
    loss_factor: float
    valve_time: float
    valve_duration: float


def read_input(path: str | Path) -> InputData:
    rows = [(line, numbers(line)) for line in _lines(path)]
    test_rows = [v for _, v in rows if len(v) == 11 and int(v[0]) in range(1, 10)]
    if len(test_rows) < 9:
        raise ValueError("Не знайдено 9 наборів випробувань")

    short = [v for _, v in rows[:35] if v]
    settings = next(v for v in short if len(v) == 4 and v[2] < 0.1)
    snapshots_row = next(v for v in short if len(v) == 9 and v[:8] == list(map(float, range(1, 9))))
    snapshots = snapshots_row[:8]
    settings_i = short.index(settings)
    heat, direction, test = (int(short[settings_i + i][0]) for i in (2, 3, 4))
    case = next(v for v in test_rows if int(v[0]) == test)
    after_cases = rows.index(next(item for item in rows if item[1] is case)) + 1
    scalars = [v[0] for _, v in rows[after_cases:60] if len(v) == 1]
    if len(scalars) < 7:
        raise ValueError("Неповний блок параметрів середовища/клапана")
    return InputData(
        *settings, tuple(snapshots), heat, direction, test,
        case[1], case[2], case[3], case[4], case[5], case[6], case[7], case[8],
        int(case[9]), case[10], *scalars[:7]
    )


@dataclass(frozen=True)
class Grid:
    x: tuple[float, ...]
    y: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class PropertyTables:
    density: Grid
    heat_capacity: Grid
    conductivity: Grid
    viscosity: Grid
    surface_t: tuple[float, ...]
    surface: tuple[float, ...]
    saturation_p: tuple[float, ...]
    latent_heat: tuple[float, ...]
    saturation_t: tuple[float, ...]
    steel: tuple[tuple[float, float, float, float], ...]
    copper: tuple[tuple[float, float, float, float], ...]


def read_properties(path: str | Path) -> PropertyTables:
    rows = [numbers(line) for line in _lines(path)]
    pos = 0

    def find(width: int, start: int) -> int:
        for i in range(start, len(rows)):
            if len(rows[i]) == width:
                return i
        raise ValueError(f"Не знайдено рядок із {width} числами")

    grids: list[Grid] = []
    for _ in range(4):
        pos = find(30, pos)
        pressures = tuple(rows[pos]); pos += 1
        data = []
        while len(data) < 53:
            pos = find(31, pos)
            data.append(rows[pos]); pos += 1
        grids.append(Grid(tuple(r[0] for r in data), pressures,
                          tuple(tuple(r[1:]) for r in data)))

    pos = find(2, pos); surface_rows = []
    while len(surface_rows) < 39:
        pos = find(2, pos); surface_rows.append(rows[pos]); pos += 1
    pos = find(3, pos); sat_rows = []
    while len(sat_rows) < 30:
        pos = find(3, pos); sat_rows.append(rows[pos]); pos += 1
    pos = find(4, pos); steel = []
    while len(steel) < 3:
        pos = find(4, pos); steel.append(tuple(rows[pos])); pos += 1
    pos = find(4, pos); copper = []
    while len(copper) < 16:
        pos = find(4, pos); copper.append(tuple(rows[pos])); pos += 1
    return PropertyTables(
        *grids,
        tuple(r[0] for r in surface_rows), tuple(r[1] for r in surface_rows),
        tuple(r[0] for r in sat_rows), tuple(r[1] for r in sat_rows),
        tuple(r[2] for r in sat_rows), tuple(steel), tuple(copper)
    )
