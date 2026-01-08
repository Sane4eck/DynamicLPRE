from __future__ import annotations
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYS_DIR = ROOT / "core" / "systems"

def write_stub(modname: str, out_path: Path):
    m = importlib.import_module(modname)

    names = []
    for k, v in m.__dict__.items():
        if k.startswith(("I_", "A_", "P_")) and isinstance(v, int):
            names.append(k)
    names.sort()

    lines = [
        "from typing import Final, Tuple\n",
        "Y_ORDER: Final[Tuple[str, ...]]\n",
        "AUX_ORDER: Final[Tuple[str, ...]]\n",
        "NY: Final[int]\n",
        "NAUX: Final[int]\n",
    ]
    for k in names:
        lines.append(f"{k}: Final[int]\n")

    out_path.write_text("".join(lines), encoding="utf-8")

def main():
    for py in SYS_DIR.glob("sys_*.py"):
        modname = f"core.systems.{py.stem}"
        out = py.with_suffix(".pyi")
        write_stub(modname, out)
        print("wrote", out)

if __name__ == "__main__":
    main()
