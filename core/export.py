import os
from openpyxl import Workbook

def save_to_excel(res, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "data"

    headers = [
        "time_s",
        "m01","m12","m13",
        "p1_Pa","p1_bar",
        "p0_Pa","p0_bar",
        "dm01","dm12","dm13","dp1",
    ]
    ws.append(headers)

    t, y, dy, aux = res.t, res.y, res.dy, res.aux
    for i in range(len(t)):
        p1_pa = float(y[i, 3])
        p0_pa = float(aux[i, 0])
        ws.append([
            float(t[i]),
            float(y[i,0]), float(y[i,1]), float(y[i,2]),
            p1_pa, p1_pa*1e-5,
            p0_pa, p0_pa*1e-5,
            float(dy[i,0]), float(dy[i,1]), float(dy[i,2]), float(dy[i,3]),
        ])

    wb.save(filepath)
