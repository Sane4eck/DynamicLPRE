import matplotlib.pyplot as plt

def plot_results(result):
    t   = [r["time"] for r in result]
    p0  = [r["p0"]   for r in result]
    p1  = [r["p1"]   for r in result]
    m01 = [r["m01"]  for r in result]
    m12 = [r["m12"]  for r in result]
    m13 = [r["m13"]  for r in result]

    # p1
    plt.figure()
    plt.plot(t, p0, label="p0")
    plt.plot(t, p1, label="p1")
    plt.xlabel("t, s")
    plt.ylabel("p1, bar")
    plt.grid()
    plt.legend()

    # m01, m12, m13
    plt.figure()
    plt.plot(t, m01, label="m01")
    plt.plot(t, m12, label="m12")
    plt.plot(t, m13, label="m13")
    plt.xlabel("t, s")
    plt.ylabel("m")
    plt.grid()
    plt.legend()

    plt.show()
