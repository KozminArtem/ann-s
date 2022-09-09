import numpy as np
import scipy.integrate
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.optimize import curve_fit
import math


N = np.array([2.5, 9.7, 97, 500, 954, 10000])
OAD_Pyro = np.array([1.57646E-4, 7.22773E-4, 0.00331, 0.01845, 0.03035, 0.13608])
z = np.polyfit(OAD_Pyro[:5], N[:5], 1)
N_poly = np.poly1d(z)
print("Concentration is equal to", N_poly(0.032), "ppm")



N = N/10000

print("OAD_Pyro")
print(OAD_Pyro)






def func_theory(x, Amp, alpha):
    return Amp*(1.0-np.exp(-alpha*x))

popt, pcov = curve_fit(func_theory, N, OAD_Pyro)

Amp,alpha = popt

print('func')
print(Amp,alpha)



point = np.arange(0, 2, 0.0001)
ans = func_theory(point, Amp, alpha)
print('Fit')
print(ans)


plt.subplots(sharey = True, figsize = (12, 10))
plt.plot(N, OAD_Pyro,'o')
plt.plot(point, ans)


plt.subplots(sharey = True, figsize = (12, 10))
div = OAD_Pyro - func_theory(N, Amp, alpha)
plt.plot(N, div)


plt.subplots(sharey = True, figsize = (12, 10))
div_persent = div/OAD_Pyro*100
plt.plot(div_persent)





















N = np.array([2.5, 9.7, 97, 500, 954])
OAD_Pyro = np.array([1.57646E-4, 7.22773E-4, 0.00331, 0.01845, 0.03035])



N = N/10000

print("OAD_Pyro")
print(OAD_Pyro)






# def func_pol_1(x, Amp):
#     return Amp*x

# popt, pcov = curve_fit(func_pol_1, N, OAD_Pyro)

# Amp = popt

# print('func')
# print(Amp)



# point = np.arange(0, 2, 0.0001)
# ans = func_pol_1(point, Amp)
# print('Fit')
# print(ans)


# plt.subplots(sharey = True, figsize = (12, 10))
# plt.plot(N, OAD_Pyro,'o')
# plt.plot(point, ans)


# plt.subplots(sharey = True, figsize = (12, 10))
# div = OAD_Pyro - func_pol_1(N, Amp)
# plt.plot(N, div)


# plt.subplots(sharey = True, figsize = (12, 10))
# div_persent = div/OAD_Pyro*100
# plt.plot(div_persent)











# def func_pol_2(x, Amp, alpha):
#     return Amp*(x - alpha*x*x)

# popt, pcov = curve_fit(func_pol_2, N, OAD_Pyro)

# Amp,alpha = popt

# print('pol_2')
# print(Amp,alpha)



# point = np.arange(0, 2, 0.0001)
# ans = func_pol_2(point, Amp,alpha)
# print('Fit')
# print(ans)


# plt.subplots(sharey = True, figsize = (12, 10))
# plt.plot(N, OAD_Pyro,'o')
# plt.plot(point, ans)


# plt.subplots(sharey = True, figsize = (12, 10))
# div = OAD_Pyro - func_pol_2(N, Amp,alpha)
# plt.plot(N, div)


# plt.subplots(sharey = True, figsize = (12, 10))
# div_persent = div/OAD_Pyro*100
# plt.plot(div_persent)


























# def func_theory_2(x, Amp, alpha, Cons):
#     return Amp*(1.0-np.exp(-alpha*x)) + Cons

# popt, pcov = curve_fit(func_theory_2, N, OAD_Pyro)

# Amp,alpha,Cons = popt
# print('func + Cons')
# print(Amp,alpha,Cons)



# point = np.arange(0, 2, 0.0001)
# ans = func_theory_2(point, Amp, alpha, Cons)
# print('Fit')
# print(ans)


# plt.subplots(sharey = True, figsize = (12, 10))
# plt.plot(N, OAD_Pyro,'o')
# plt.plot(point, ans)


# plt.subplots(sharey = True, figsize = (12, 10))
# div = OAD_Pyro - func_theory_2(N, Amp, alpha, Cons)
# plt.plot(N, div)


# plt.subplots(sharey = True, figsize = (12, 10))
# div_persent = div/OAD_Pyro*100
# plt.plot(div_persent)









# def func_theory_3(x, Amp, alpha, Cons, sdv):
#     return Amp*(1.0-np.exp(-alpha*x + sdv)) + Cons

# popt, pcov = curve_fit(func_theory_3, N, OAD_Pyro)

# Amp,alpha,Cons,sdv = popt
# print('func + Cons + sdv')
# print(Amp,alpha,Cons,sdv)



# point = np.arange(0, 2, 0.0001)
# ans = func_theory_3(point, Amp, alpha, Cons,sdv)
# print('Fit')
# print(ans)


# plt.subplots(sharey = True, figsize = (12, 10))
# plt.plot(N, OAD_Pyro,'o')
# plt.plot(point, ans)


# plt.subplots(sharey = True, figsize = (12, 10))
# div = OAD_Pyro - func_theory_3(N, Amp, alpha, Cons,sdv)
# plt.plot(N, div)


# plt.subplots(sharey = True, figsize = (12, 10))
# div_persent = div/OAD_Pyro*100
# plt.plot(div_persent)





plt.show()