import numpy as np
import math
import matplotlib.pyplot as plt

n = None
t = None
h = None
v = None
volvol = None
k = None


def heston(rand):
    vol = np.zeros(len(rand))
    vol[0] = v
    for i in range(1, len(vol)):
        vol[i] = vol[i - 1] + k * (v - vol[i - 1]) * h + volvol * math.sqrt(vol[i - 1]) * rand[i] * h
        if vol[i] <= 0.0:
            print("volatility dropped to zero at step",i)
            print(vol)
            raise Exception
    return vol

def heston_v2(rand):
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
    hstep = np.zeros(len(vol))
    hstep[0] = h
    for i in range(1,len(vol)):
        hstep[i] = (vol[i-1]*math.sqrt(hstep[i-1])/vol[i])**2
    return hstep

def normalize(vol, hstep, time):
    c_time = sum(hstep)
    c = time/c_time
    hstep *= c
    return vol, hstep

def gen_hest_vol(nt,tt,vt,kt,volvolt,norm=True,adjust_step=False):
    # Locking random seed to provide stability during development
    np.random.seed(42)

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
        rand = np.random.normal(loc=0, scale=1., size=(round(n*5)))
        vol = heston_v2(rand)
        hstep = calc_hstep(vol)
        i = 0
        ts = 0
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
    rand = np.random.normal(loc=0, scale=1., size=(n))
    vol = heston_v2(rand)
    hstep = calc_hstep(vol)
    if norm:
        return normalize(vol,hstep,t)
    return vol, hstep

if __name__ == "__main__":
    #np.random.seed(42)


    # delete later on
    n = 100
    t = 1
    h = t / n
    v = 0.25
    k = 0.2
    volvol= 2

#        vol,h = gen_hest_vol(n,t,v,k,volvol,norm=False,adjust_step=False)

    """
    d = np.zeros(len(h))
    d[0]= h[0]
    for i in range(1,len(d)):
        d[i] = d[i-1] + h[i]

    plt.figure(0)
    plt.plot(vol)
    plt.figure(1)
    plt.plot(d,vol)
    plt.show()
"""

"""
plt.plot(vol)
plt.show()

plt.plot(hstep)
plt.show()
"""