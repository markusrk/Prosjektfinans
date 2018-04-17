import numpy as np
import math
import matplotlib.pyplot as plt


def jarrow_rudd(s, k, t, v, rf, cp, div=0.0, am=False, n=100,x=1,b=0):
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
    x : Multiplier for payoff
    b : constant payoff bonus
  """

    # Basic calculations
    h = t / n
    u = math.exp((rf-div)*h +v * math.sqrt(h))
    d = math.exp((rf-div)*h-v * math.sqrt(h))
    drift = math.exp((rf) * h)
    stkdrift = math.exp((rf-div)*h)
    q = (stkdrift - d) / (u - d)



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
        if am:
            if optval[n, j] < cp * (stkval[n, j]*x+b - k):
                sellflag[i, j] = 1
        optval[n, j] = max(0, cp * (stkval[n, j]*x+b - k))
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            optval[i, j] = (q * optval[i + 1, j] + (1 - q) * optval[i + 1, j + 1]) / drift
            if am:
                if optval[i, j] < cp * (stkval[i, j]*x+b - k):
                    sellflag[i, j] = 1
                optval[i, j] = max(optval[i, j], cp * (stkval[i, j]*x+b - k))

    return {"value": optval[0, 0], "sellflag": sellflag}


def plotsellflags(func):
    plt.matshow(func)
    plt.xlabel("Downs")
    plt.ylabel("Periods")
    plt.show()

def plotexerciseboundry(var, highest,s, k, t, rf, cp, div=0.0, am=False, n=100):
    """ Finds and plots the highest/lowest value at which the option will be excercised"""

    h = t / n

    # changes time and periods to allow for nice print
    t = t+1
    dn = round(n*t/(t-1))-n
    n += dn
    for v in var:
        u = math.exp((rf - div) * h + v * math.sqrt(h))
        d = math.exp((rf - div) * h - v * math.sqrt(h))
        sellflag = jarrow_rudd(s, k, t, v, rf, cp, div=div, am=am, n=n)["sellflag"]

        if highest == True:
            boundry = sellflag.argmax(axis=1)
        else:
            boundry = n-np.flip(sellflag,axis=1).argmax(axis=1)
            boundry[boundry>=n]=0
        boundryprice = []
        for i in range(len(boundry)):
            boundryprice.append(s*u**(i-boundry[i])*d**(boundry[i]))
        plt.plot(np.linspace(0,t,num=n-dn),boundryprice[dn+1:])
    plt.show()


if __name__ == "__main__":
    #print(jarrow_rudd(100.0, 100.0, 1.0, 0.3, 0.03, 1, div=0.0, am=False, n=100)["value"])
    #print(jarrow_rudd(100.0, 100.0, 1.0, 0.3, 0.03, 1, div=0.5, am=True, n=100)["value"])
    #plotsellflags(jarrow_rudd(41.0, 40.0, 1.0, 0.3, 0.08, -1, div=0.00, am=True, n=3)["sellflag"])
    #plotsellflags(jarrow_rudd(41.0, 40.0, 1.0, 0.3, 0.08, 1, div=0.00, am=True, n=3)["sellflag"])
    #plotsellflags(jarrow_rudd(41.0, 40.0, 2.0, 0.3, 0.08, 1, div=0.00, am=True, n=2)["sellflag"])
    #plotsellflags(jarrow_rudd(110.0, 100.0, 1.0, 0.3, 0.05, 1, div=0.035, am=True, n=3)["sellflag"])
    #plotsellflags(jarrow_rudd(1.05, 1.10, 0.5, 0.1, 0.055, -1, div=0.031, am=True, n=3)["sellflag"])
    #plotsellflags(jarrow_rudd(100.0, 100.0, 1.0, 0.5, 0.03, 1, div=0.04, am=True, n=100)["sellflag"])
    #plotsellflags(jarrow_rudd(100.0, 100.0, 1.0, 0.7, 0.03, 1, div=0.04, am=True, n=100)["sellflag"])

    #plotsellflags(jarrow_rudd(100.0, 100.0, 5.0, 0.5, 0.05, 1, div=0.05, am=True, n=500)["sellflag"])
    #plotsellflags(jarrow_rudd(100.0, 100.0, 5.0, 0.5, 0.05, -1, div=0.05, am=True, n=500)["sellflag"])
    #plotexerciseboundry([0.5,0.3,0.1],True,100.0, 100.0, 5.0, 0.05, -1, div=0.02, am=True, n=500)
    plotexerciseboundry([0.5, 0.3, 0.1], True, 100.0, 100.0, 5.0, 0.05, -1, div=0.05, am=True, n=500)
    plotexerciseboundry([0.5, 0.3, 0.1], False, 100.0, 100.0, 5.0, 0.05, 1, div=0.05, am=True, n=500)
    print("Done")

""" s, k, t, v, rf, cp, div=0.0, am=False, n=100  """