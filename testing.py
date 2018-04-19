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
    vol = np.zeros(n)
    vol[0] = v
    for i in range(1, len(vol)):
        vol[i] = vol[i - 1] + k * (v - vol[i - 1]) * h + volvol * math.sqrt(vol[i - 1]) * rand[i] * h
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
    vol = vol*c
    hstep *= c
    return vol, hstep

def gen_hest_vol(nt,tt,vt,kt,volvolt,norm=False):
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


    rand = np.random.normal(loc=0, scale=1., size=(n))
    vol = heston(rand)
    hstep = calc_hstep(vol)
    if norm:
        return normalize(vol,hstep,t)
    return vol, hstep

if __name__ == "__main__":
    np.random.seed(42)

    # delete later on
    n = 500
    t = 1
    h = t / n
    v = 0.7
    k = 1

    rand = np.random.normal(loc=0, scale=1., size=(n))


    vol = heston(rand)
    hstep = calc_hstep(vol)
    vol,hstep = normalize(vol,hstep,t)
    up = np.exp(vol*np.sqrt(hstep))
    down = np.exp(-vol*np.sqrt(hstep))

    for i in range(50,100):
        print(up[i]*down[i+1])
        print(down[i]*up[i+1])

    plt.plot(up)
    plt.plot(down)
    plt.show()


"""
plt.plot(vol)
plt.show()

plt.plot(hstep)
plt.show()
"""