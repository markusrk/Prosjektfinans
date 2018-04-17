import numpy as np
import math
import matplotlib.pyplot as plt
from binomial_v5 import jarrow_rudd

# Initialize variables
x = np.linspace(20,60,40)
t0 = np.linspace(-20,20,40)
t0[t0<0] = 0
t03 = np.zeros(40)
t06 = np.zeros(40)
t1 = np.zeros(40)
price = 6.285

# Calculate payoffs
for i in range(len(t03)):
    t03[i]=jarrow_rudd(s=x[i],k=40,t=0.3,v=0.3,rf=0.08,cp=1,div=0.,am=True,n=50)["value"]
    t06[i]=jarrow_rudd(s=x[i],k=40,t=0.6,v=0.3,rf=0.08,cp=1,div=0.,am=True,n=50)["value"]
    t1[i]=jarrow_rudd(s=x[i],k=40,t=1,v=0.3,rf=0.08,cp=1,div=0.,am=True,n=50)["value"]

t = [t0,t03,t06,t1,x]

# plot payoff values
for tp in t:
    plt.plot(x,tp)
#plt.ylim((0,25))
plt.show()

# plot profit values
t0 -= price*np.exp(0.08)
t03 -= price*np.exp(0.08*9/12)
t06 -= price*np.exp(0.08*6/12)
t1 -= price*np.exp(0.08*1/365)

for tp in t:
    plt.plot(x,tp)
plt.axhline(y=0,color="black",linestyle="-")
#plt.ylim((-10,25))
plt.show()