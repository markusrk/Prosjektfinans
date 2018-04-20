import numpy as np
import math
import matplotlib.pyplot as plt
from testing import gen_hest_vol

def jarrow_rudd(s, k, t, v, rf, cp, div=0.0, am=False, n=100,x=1,b=0,kv=0,volvol=0,adjust_step=False):
    """ Price an option using the Jarrow-Rudd binomial model

    Parameters
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
        kv : dampening constant for volatility function
        volvol : volatility of the volatility

    Returns
        value : the value of the option at starting time
        sellflag : 2d array showing which states the option should be excercised in
  """

    # Basic calculations
    vol, h = gen_hest_vol(n,t,v,kv,volvol,norm=False,adjust_step=adjust_step)
    n = len(h)
    u = np.exp((rf-div)*h +vol * np.sqrt(h))
    d = np.exp((rf-div)*h-vol * np.sqrt(h))
    drift = np.exp((rf) * h)
    stkdrift = np.exp((rf-div)*h)
    q = (stkdrift - d) / (u - d)



    # Process the terminal stock price and dividend amounts
    stkval = np.zeros((n + 1, n + 1))
    optval = np.zeros((n + 1, n + 1))
    sellflag = np.zeros((n + 1, n + 1))
    dividend = np.zeros((n + 1, n + 1))
    stkval[0, 0] = s
    for i in range(1, n + 1):
        stkval[i, 0] = stkval[i - 1, 0] * u[i-1]
        for j in range(1, i + 1):
            stkval[i, j] = stkval[i - 1, j - 1] * d[i-1]



    # Backward recursion for option price
    # The first for loop sets the option value in the final row, the second for loop fills the rest of the table
    for j in range(n + 1):
        if am:
            if optval[n, j] < cp * (stkval[n, j]*x+b - k):
                sellflag[i, j] = 1
        optval[n, j] = max(0, cp * (stkval[n, j]*x+b - k))
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            optval[i, j] = (q[i] * optval[i + 1, j] + (1 - q[i]) * optval[i + 1, j + 1]) / drift[i]
            if am:
                if optval[i, j] < cp * (stkval[i, j]*x+b - k):
                    sellflag[i, j] = 1
                optval[i, j] = max(optval[i, j], cp * (stkval[i, j]*x+b - k))


    # Calculating excercise price
    if cp==-1:
        boundry = sellflag.argmax(axis=1)
    else:
        boundry = n - np.flip(sellflag, axis=1).argmax(axis=1)
        boundry[boundry >= n] = 0
    boundryprice = []
    for i in range(len(boundry)):
        boundryprice.append(s * u[1] ** (i - boundry[i]) * d[1] ** (boundry[i]))

    return {"value": optval[0, 0], "sellflag": sellflag, "boundryprice": boundryprice, "h": h}




def plotexerciseboundry(var, highest,s, k, t, rf, cp, div=0.0, am=False, n=100,x=1,b=0,kv=0,volvol=0,adjust_step=False):
    """ Finds and plots the highest/lowest value at which the option will be excercised"""

    h = t / n

    # Changes time and periods to allow the tree to
    # expand before the plotting period so we are sure the excercise points are available
    t = t+1.5
    dn = round(n*t/(t-1.5))-n
    n += dn

    # Calculates the up and down movements, pulls the highest/lowest
    # value for excercise (depending on whether we have a call or a put)
    for v in var:
        boundryprice = jarrow_rudd(s, k, t, v, rf, cp, div=div, am=am, n=n,x=x,b=b,kv=kv,volvol=volvol,adjust_step=adjust_step)["boundryprice"]
        plt.plot(np.linspace(0,t-1,num=len(boundryprice)-dn-1), boundryprice[dn+1:], label="var= "+str(v))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #print(jarrow_rudd(100.0, 100.0, 1.0, 0.3, 0.03, 1, div=0.0, am=False, n=100)["value"])
    #print(jarrow_rudd(100.0, 100.0, 1.0, 0.3, 0.03, 1, div=0.5, am=True, n=100)["value"])
    #plotsellflags(jarrow_rudd(41.0, 40.0, 1.0, 0.3, 0.08, -1, div=0.00, am=True, n=3)["sellflag"])
    #plotsellflags(jarrow_rudd(41.0, 40.0, 1.0, 0.3, 0.08, 1, div=0.00, am=True, n=1)["sellflag"])
    #plotsellflags(jarrow_rudd(41.0, 40.0, 2.0, 0.3, 0.08, 1, div=0.00, am=False, n=2)["sellflag"])
    #plotsellflags(jarrow_rudd(110.0, 100.0, 1.0, 0.3, 0.05, 1, div=0.035, am=True, n=3)["sellflag"])
    #plotsellflags(jarrow_rudd(1.05, 1.10, 0.5, 0.1, 0.055, -1, div=0.031, am=True, n=3)["sellflag"])
    #plotsellflags(jarrow_rudd(100.0, 100.0, 1.0, 0.5, 0.03, 1, div=0.04, am=True, n=100)["sellflag"])
    #plotsellflags(jarrow_rudd(100.0, 100.0, 1.0, 0.7, 0.03, 1, div=0.04, am=True, n=100)["sellflag"])
    #print(jarrow_rudd(100.0, 100.0, 5.0, 0.25, 0.02, 1, div=0.7, am=True, n=500)["value"])
    #print(jarrow_rudd(100.0, 100.0, 5.0, 0.25, 0.02, 1, div=0.07, am=True, n=500)["value"])

    #plotsellflags(jarrow_rudd(100.0, 100.0, 5.0, 0.5, 0.05, 1, div=0.05, am=True, n=500)["sellflag"])
    #plotsellflags(jarrow_rudd(100.0, 100.0, 5.0, 0.5, 0.05, -1, div=0.05, am=True, n=500)["sellflag"])
    #plotexerciseboundry([0.5,0.3,0.1],True,100.0, 100.0, 5.0, 0.05, -1, div=0.02, am=True, n=500)

    """Answers for problem 2a"""
    #plotexerciseboundry([0.5, 0.3, 0.1], True, 100.0, 100.0, 5.0, 0.02, -1, div=0.07, am=True, n=500)
    #plotexerciseboundry([0.5, 0.3, 0.1], False, 100.0, 100.0, 5.0, 0.02, 1, div=0.07, am=True, n=500)


    """Answers for problem 2b"""
    #print("Value of option: ", jarrow_rudd(s=100.0, k=00.0, t=5.0, v=0.25, rf=0.02, cp=1, div=0.07, am=True, n=500,x=1.5,b=-200)["value"])

    """Answer for question 3a"""
    plotexerciseboundry([0.5, 0.3, 0.1],  True, 100.0, 100.0, 5.0, 0.02, -1, div=0.07, am=True, n=500, x=1, b=0,volvol=0.8, kv=0.2,adjust_step=True)
    plotexerciseboundry([0.5, 0.3, 0.1], False, 100.0, 100.0, 5.0, 0.02,  1, div=0.07, am=True, n=500, x=1, b=0,volvol=0.8, kv=0.2,adjust_step=True)
    #print("Value of option: ", jarrow_rudd(s=100.0, k=00.0, t=5.0, v=0.25, rf=0.02, cp=1, div=0.07, am=True, n=500,x=1.5,b=-200,volvol=2,kv=0.2,adjust_step=True)["value"])
    r = 0
    n = 20
    res = np.zeros(n)
    for x in range(n):
        res[x] = jarrow_rudd(s=100.0, k=00.0, t=5.0, v=0.25, rf=0.02, cp=1, div=0.07,
                            am=True, n=500,x=1.5,b=-200,volvol=0.2,kv=0.2,adjust_step=True)["value"]
    print("avg. value: ",sum(res)/len(res))
    print("st.dev: ", np.std(res))


""" s, k, t, v, rf, cp, div=0.0, am=False, n=100,x=1,b=0):

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