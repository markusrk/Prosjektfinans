import numpy as np
import math
import matplotlib.pyplot as plt


def jarrow_rudd(s, k, t, v, rf, cp, div=0.0, am=False, n=100):
    """ Price an option using the Jarrow-Rudd binomial model

    s :  initial stock price
    k : strike price
    t : expiration time
    v :  volatility
    rf : risk-free rate
    cp : +1/-1 for call /put
    div : Dividend amount annual percentage
    am : True/False for American/European
    n : binomial steps
  """

    # Basic calculations
    h = t / n
    u = math.exp((rf - 0.5 * math.pow(v, 2)) * h + v * math.sqrt(h))
    d = math.exp((rf - 0.5 * math.pow(v, 2)) * h - v * math.sqrt(h))
    drift = math.exp((rf-div) * h)
    q = (drift - d) / (u - d)

    # Process the terminal stock price and dividend amounts
    stkval = np.zeros((n + 1, n + 1))
    optval = np.zeros((n + 1, n + 1))
    sellflag = np.zeros((n + 1, n + 1))
    dividend = np.zeros((n + 1, n + 1))
    stkval[0, 0] = s
    for i in range(1, n + 1):
        stkval[i, 0] = stkval[i - 1, 0] * u
        for j in range(1, i + 1):
            stkval[i, j] = stkval[i - 1, j - 1] * d



    # Backward recursion for option price
    for j in range(n + 1):
        optval[n, j] = max(0, cp * (stkval[n, j] - k))
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            optval[i, j] = (q * optval[i + 1, j] + (1 - q) * optval[i + 1, j + 1]) / drift
            if am:
                if optval[i, j] < cp * (stkval[i, j] - k):
                    sellflag[i, j] = 1
                optval[i, j] = max(optval[i, j], cp * (stkval[i, j] - k))

    return {"value": optval[0, 0], "sellflag": sellflag}


def plotsellflags(func):
    plt.matshow(func)
    plt.xlabel("Downs")
    plt.ylabel("Periods")
    plt.show()

if __name__ == "__main__":
    print(jarrow_rudd(100.0, 100.0, 1.0, 0.3, 0.03, 1, div=0.5, am=False, n=100)["value"])
    print(jarrow_rudd(100.0, 100.0, 1.0, 0.3, 0.03, 1, div=0.5, am=True, n=100)["value"])
    plotsellflags(jarrow_rudd(100.0, 100.0, 1.0, 0.3, 0.03, 1, div=0.04, am=True, n=100)["sellflag"])

    print("Done")

