import numpy as np
import matplotlib.pyplot as plt
from timeVol import gen_hest_vol
from binomial import binomial, plotexerciseboundry

"""Answers for problem 2a"""
# Plot put
plotexerciseboundry([0.5, 0.25, 0.1], 100.0, 100.0, 5.0, 0.02, -1, div=0.07, am=True, n=500)
# Plot call
plotexerciseboundry([0.5, 0.25, 0.1], 100.0, 100.0, 5.0, 0.02, 1, div=0.07, am=True, n=500)

"""Answers for problem 2b"""
# Print value of option
print("Value of option: ",
      binomial(s=100.0, k=00.0, t=5.0, v=0.25, rf=0.02, cp=1, div=0.07, am=True, n=500, x=1.5, b=-200)["value"])

"""Answers for problem 2c"""
# Initialize variables
steps = 20
x = np.linspace(80, 160, steps)
t0 = 1.5 * x - 200
t0[t0 < 0] = 0
t03 = np.zeros(steps)
t06 = np.zeros(steps)
t1 = np.zeros(steps)

# Calculate payoffs
for i in range(steps):
    t03[i] = binomial(s=x[i], k=0, t=0.25, v=0.25, rf=0.02, cp=1, div=0.07, am=True, n=200, x=1.5, b=-200)["value"]
    t06[i] = binomial(s=x[i], k=0, t=1, v=0.25, rf=0.02, cp=1, div=0.07, am=True, n=200, x=1.5, b=-200)["value"]
    t1[i] = binomial(s=x[i], k=00.0, t=5.0, v=0.25, rf=0.02, cp=1, div=0.07, am=True, n=200, x=1.5, b=-200)["value"]

# Plot payoff values
plt.plot(x, t03, label="0.25 years")
plt.plot(x, t06, label="1 year")
plt.plot(x, t1, label="5 years")
plt.plot(x, t0, label="Expiration")
plt.legend()
plt.show()

"""Answer for problem 3a"""
# Plot put
plotexerciseboundry([0.5, 0.25, 0.1], 100.0, 100.0, 5.0, 0.02, -1, div=0.07, am=True, n=500, x=1, b=0, volvol=0.8,
                    kv=0.2, adjust_step=True)
# Plot call
plotexerciseboundry([0.5, 0.25, 0.1], 100.0, 100.0, 5.0, 0.02, 1, div=0.07, am=True, n=500, x=1, b=0, volvol=0.8,
                    kv=0.2, adjust_step=True)
# Print option value
r = 0
n = 20
res = np.zeros(n)
for x in range(n):
    res[x] = binomial(s=100.0, k=00.0, t=5.0, v=0.25, rf=0.02, cp=1, div=0.07,
                      am=True, n=500, x=1.5, b=-200, volvol=0.8, kv=0.2, adjust_step=False)["value"]
print("avg. value: ", sum(res) / len(res))
print("st.dev: ", np.std(res))

"""Answer for problem 3b"""
# Calculate option value sample
volvol = np.linspace(0., 1, 200)
res = np.zeros((len(volvol)))

for x in range(len(volvol)):
    res[x] = binomial(s=100.0, k=100.0, t=1, v=0.25, rf=0.02, cp=1, div=0.07,
                      am=True, n=100, x=1, b=00, volvol=volvol[x], kv=0.8, adjust_step=True)["value"]
# Fit regression line
z = np.polyfit(volvol, res, 1)
p = np.poly1d(z)

# Plot graph and print regression values
plt.plot(volvol, res, "ro")
plt.plot(volvol, p(volvol))
plt.xlim(0, 1)
plt.xlabel("$\\bar \sigma$")
plt.ylabel("Payoff")
print("y=%.6fx+(%.6f)" % (z[0], z[1]))
plt.show()

"""Code for making volatility variation plot"""
# Initialize variables
n = 100
t = 5
h = t / n
v = 0.25
k = 0.2
volvol = 1.1

# generate volatility and step size vectors.
vol, h = gen_hest_vol(n, t, v, k, volvol, norm=False, adjust_step=False)
d = np.zeros(len(h))
d[0] = h[0]
for i in range(1, len(d)):
    d[i] = d[i - 1] + h[i]

n = 10000
vol2, h2 = gen_hest_vol(n, t, v, k, volvol, norm=False, adjust_step=False)
d2 = np.zeros(len(h2))
d2[0] = h2[0]
for i in range(1, len(d2)):
    d2[i] = d2[i - 1] + h2[i]

# Print graph showing all 4 plots
fig = plt.figure()
p1 = fig.add_subplot(2, 2, 1)
p2 = fig.add_subplot(2, 2, 2)
p3 = fig.add_subplot(2, 2, 3)
p4 = fig.add_subplot(2, 2, 4)

p1.plot(vol)
p1.set_title(label="constant h")
p1.set_ylabel(ylabel=" n = 100")
p2.plot(d, vol)
p2.set_title(label="adjusted h")
p3.plot(vol2)
p3.set_ylabel(ylabel=" n = 10 000")
p4.plot(d2, vol2)
plt.show()