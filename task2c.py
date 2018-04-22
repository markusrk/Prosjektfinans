import numpy as np
import math
import matplotlib.pyplot as plt
from binomial_v5 import jarrow_rudd

# parameters
steps = 20

# Initialize variables
x = np.linspace(80,160,steps)
t0 = 1.5*x-200
t0[t0<0] = 0
t03 = np.zeros(steps)
t06 = np.zeros(steps)
t1 = np.zeros(steps)

# Calculate payoffs
for i in range(steps):
    t03[i]=jarrow_rudd(s=x[i],k=0,t=0.25,v=0.25,rf=0.02,cp=1,div=0.07,am=True,n=200,x=1.5,b=-200)["value"]
    t06[i]=jarrow_rudd(s=x[i],k=0,t=1, v=0.25,rf=0.02,cp=1,div=0.07,am=True,n=200,x=1.5,b=-200)["value"]
    t1[i] =jarrow_rudd(s=x[i], k=00.0, t=5.0, v=0.25, rf=0.02, cp=1, div=0.07, am=True, n=200,x=1.5,b=-200)["value"]


# plot payoff values
plt.plot(x,t03,label="0.25 years")
plt.plot(x,t06,label="1 year")
plt.plot(x,t1,label="5 years")
plt.plot(x,t0,label="Expiration")
plt.legend()
plt.show()


# plot explanation graphs
xp=[x[0],x[(steps-1)],x[round((steps-1)/6)],x[round((steps-1)*5/6)],x[round((steps-1)*2/6)],x[round((steps-1)*4/6)]]
yp=[t06[0],t06[(steps-1)],t06[round((steps-1)/6)],t06[round((steps-1)*5/6)],t06[round((steps-1)*2/6)],t06[round((steps-1)*4/6)]]


def connectpoints(x,y,p1,p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x1,x2],[y1,y2],'k-')

connectpoints(xp,yp,0,1)
connectpoints(xp,yp,2,3)
connectpoints(xp,yp,4,5)

plt.plot(x,t06,label="1 year")
plt.plot(xp,yp, 'ro')
plt.show()
