import numpy as np
import matplotlib.pyplot as plt
from timeVar import gen_hest_var

def binomial(s, k, t, v, rf, cp, div=0.0, am=False, n=100, x=1, b=0, kv=0, volvol=0, adjust_step=False):
    """ Calculates the price of an option using the binomial model

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
        boundryprice : boundryprice to sell the option
        h : time step vector
  """

    # Calculate variables for later use
    vol, h = gen_hest_var(n, t, v, kv, volvol, norm=True, adjust_step=adjust_step)
    n = len(h)
    u = np.exp((rf-div)*h +vol * np.sqrt(h))
    d = np.exp((rf-div)*h-vol * np.sqrt(h))
    drift = np.exp((rf) * h)
    stkdrift = np.exp((rf-div)*h)
    q = (stkdrift - d) / (u - d)



    # Process the terminal stock price and dividend amounts
    stock_price = np.zeros((n + 1, n + 1))
    optionvalues = np.zeros((n + 1, n + 1))
    sellflag = np.zeros((n + 1, n + 1))
    dividend = np.zeros((n + 1, n + 1))
    stock_price[0, 0] = s
    for i in range(1, n + 1):
        stock_price[i, 0] = stock_price[i - 1, 0] * u[i-1]
        for j in range(1, i + 1):
            stock_price[i, j] = stock_price[i - 1, j - 1] * d[i-1]



    # Backward recursion for option price
    # The first for loop sets the option value in the final row, the second for loop fills the rest of the table
    for j in range(n + 1):
        if am:
            if optionvalues[n, j] < cp * (stock_price[n, j]*x+b - k):
                sellflag[i, j] = 1
        optionvalues[n, j] = max(0, cp * (stock_price[n, j]*x+b - k))
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            optionvalues[i, j] = (q[i] * optionvalues[i + 1, j] + (1 - q[i]) * optionvalues[i + 1, j + 1]) / drift[i]
            if am:
                if optionvalues[i, j] < cp * (stock_price[i, j]*x+b - k):
                    sellflag[i, j] = 1
                optionvalues[i, j] = max(optionvalues[i, j], cp * (stock_price[i, j]*x+b - k))


    # Calculating excercise price
    if cp==-1:
        boundry = sellflag.argmax(axis=1)
    else:
        boundry = n - np.flip(sellflag, axis=1).argmax(axis=1)
        boundry[boundry >= n] = 0
    boundryprice = []
    for i in range(len(boundry)):
        boundryprice.append(s * u[1] ** (i - boundry[i]) * d[1] ** (boundry[i]))

    return {"value": optionvalues[0, 0], "sellflag": sellflag, "boundryprice": boundryprice, "h": h}




def plotexerciseboundry(vol, s, k, t, rf, cp, div=0.0, am=False, n=100, x=1, b=0, kv=0, volvol=0, adjust_step=False):
    """ Finds and plots the highest/lowest value at which the option will be excercised, for a set of volatilities
    Variables
        Sames as binomial, the list "vol" replaces a single volatility float.
    """

    # Changes time and periods to allow the tree to
    # expand before the plotting period so we are sure the exercise points are available
    t = t+2
    dn = round(n*t/(t-2))-n
    n += dn

    # Calculates the up and down movements, pulls the highest/lowest
    # value for exercise (depending on whether we have a call or a put)
    for v in vol:
        boundryprice = binomial(s, k, t, v, rf, cp, div=div, am=am, n=n, x=x, b=b, kv=kv, volvol=volvol, adjust_step=adjust_step)["boundryprice"]
        plt.plot(np.linspace(0,t-2,num=len(boundryprice)-dn-1), boundryprice[dn+1:], label="vol= "+str(v))
    plt.legend()
    plt.show()
