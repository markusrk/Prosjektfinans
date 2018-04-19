import numpy as np
import math
import matplotlib.pyplot as plt
from binomial_v5 import jarrow_rudd

# parameters
steps = 200

# Initialize variables
x = np.linspace(110,135,steps)
t0 = 1.5*x-200
t0[t0<0] = 0
t03 = np.zeros(steps)
t06 = np.zeros(steps)
t1 = np.zeros(steps)

# Calculate payoffs
for i in range(len(t03)):
    t03[i]=jarrow_rudd(s=x[i],k=0,t=0.3,v=0.25,rf=0.02,cp=1,div=0.7,am=True,n=50,x=1.5,b=-200)["value"]
    t06[i]=jarrow_rudd(s=x[i],k=0,t=0.6,v=0.25,rf=0.02,cp=1,div=0.7,am=True,n=50,x=1.5,b=-200)["value"]
    t1[i] =jarrow_rudd(s=x[i],k=0,t=1,  v=0.25,rf=0.02,cp=1,div=0.7,am=True,n=50,x=1.5,b=-200)["value"]


# plot payoff values
plt.plot(x,t03,label="3 months")
plt.plot(x,t06,label="6 months")
plt.plot(x,t1,label="1 year")
plt.plot(x,t0,label="1 day")
plt.legend()
plt.show()
