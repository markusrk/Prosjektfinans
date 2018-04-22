import numpy as np
import math

n = None
t = None
h = None
v = None
volvol = None
k = None


def heston_v2(rand):
    """Generates volatility vector from a string of random numbers (preferably normally distributed"""
    vol = np.zeros(len(rand))
    vol[0] = v
    for i in range(1, len(vol)):
        vol[i] = vol[i - 1] + k * (v - vol[i - 1]) * h + volvol * math.sqrt(vol[i - 1]) * rand[i] * h
        if vol[i] <= 0.1:
            n = 2
            while vol[i] <= 0.0:
                n = n*2
                dh = h/n
                tvol = np.zeros(n)
                tvol[0] = vol[i-1]
                for x in range(1,len(tvol)):
                    tvol[x] = tvol[x - 1] + k * (v - tvol[x - 1]) * dh + volvol * math.sqrt(tvol[x - 1]) * rand[x] * dh
                vol[i] = tvol[-1]
    return vol


def calc_hstep(vol):
    """Returns a vector of time steps that solves the UD=DU equation for a volatility string"""
    hstep = np.zeros(len(vol))
    hstep[0] = h
    for i in range(1,len(vol)):
        hstep[i] = (vol[i-1]*math.sqrt(hstep[i-1])/vol[i])**2
    return hstep

def normalize(vol, hstep, time):
    """Normalizes a time vector to fit the given time period"""
    c_time = sum(hstep)
    c = time/c_time
    hstep *= c
    return vol, hstep

def gen_hest_vol(nt,tt,vt,kt,volvolt,norm=True,adjust_step=False):
    """Generates volatility and time step vectors

    nt : time steps
    tt : time period in years
    kt : kappa in the heston equation
    volvol : volatility of the volatility
    norm : normalizes the timevector to fit if True
    adjust_step : add/removes step to fit time period if true

    """
    # Locking random seed to provide stability during development
    # np.random.seed(42)

    # Set global variables to input
    global n
    global t
    global v
    global k
    global volvol
    global h
    n = nt
    t = tt
    h = t/n
    v = vt
    k = kt
    volvol = volvolt

    if adjust_step:
        # Calculates volatility and timestep vectors, is delibarately made too long
        rand = np.random.normal(loc=0, scale=1., size=(round(n*5)))
        vol = heston_v2(rand)
        hstep = calc_hstep(vol)
        i = 0
        ts = 0
        # Finds number of periods to include
        while True:
            if ts < t:
                ts += hstep[i]
                i+=1
                if i >= round(n*3):
                    print("i=",i)
                    print("sum(hstep)= ",sum(hstep))
                    raise Exception()
                continue
            break
        return vol[:i],hstep[:i]
    # Calculates vectors and nomalizes length before return
    rand = np.random.normal(loc=0, scale=1., size=(n))
    vol = heston_v2(rand)
    hstep = calc_hstep(vol)
    if norm:
        return normalize(vol,hstep,t)
    return vol, hstep
