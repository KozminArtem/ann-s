from numpy import*
import matplotlib.pyplot as plt

x= arange(-4,30,0.01)

def w(a,b,t):    
    f =(1/a**0.5)*exp(-0.5*((t-b)/a)**2)* (((t-b)/a)**2-1)
    return f


plt.figure()


plt.title("Вейвлет «Мексиканская шляпа»:\n$1/\sqrt{a}*exp(-0,5*t^{2}/a^{2})*(t^{2}-1)$")
y=[w(1,12,t) for t in x]
plt.plot(x,y,label="$\psi(t)$ a=1,b=12") 
y=[w(2,12,t) for t in x]
plt.plot(x,y,label="$\psi_{ab}(t)$ a=2 b=12")   
y=[w(4,12,t) for t in x]
plt.plot(x,y,label="$\psi_{ab}(t)$ a=4 b=12")   
plt.legend(loc='best')
plt.grid(True)






from scipy.integrate import quad
from numpy import*
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

N=256
T=50

Size = 200

def S(t):
    if t >= 0 and t <=Size:
        return sin(2*pi*t/T)
    else:
        return 0
#     return sin(2*pi*t/T)



def S_noise(t):
    return sin(2*pi*t/T)+ np.random.normal(0, 0.2, 1)

def w(a,b):    
    f = lambda t :(1/a**0.5)*exp(-0.5*((t-b)/a)**2) * (((t-b)/a)**2-1)*S(t)  
    r = quad(f, -N, N)  #интеграл
    return round(r[0],3)  #round(number[, ndigits]) - округляет число number до ndigits знаков после запятой

def w_n(a,b):    
    f = lambda t :(1/a**0.5)*exp(-0.5*((t-b)/a)**2) * (((t-b)/a)**2-1)*S_noise(t)  
    r = quad(f, -N, N)  #интеграл
    return round(r[0],3)  #round(number[, ndigits]) - округляет число number до ndigits знаков после запятой



plt.figure()
plt.title(' Гармоническое колебание', size=12)
y=[S(t) for t in arange(0,Size,1)]
x=[t for t in arange(0,Size,1)]
plt.plot(x,y)
plt.grid()

Size_A = 50
A = arange(1,Size_A,1)
print(len(A))
B = arange(0, Size, 1)
print(len(B))
# z = array([w(i,j) for i in A for j in B])

Z = zeros((Size, Size_A - 1))

for i in A:
    for j in B:
        Z[j][i-1] = w(i,j)

print(Z)





A_par, B_par = meshgrid(A, B)

# B_par, A_par = meshgrid(B^)

# print(A_par, B_par)
# Z = z.reshape(Size_A - 1, Size)



print(len(Z))
print(len(Z[0]))




fig = plt.figure('Вейвлет- спектр: гармонического колебания')
ax = Axes3D(fig)
ax.plot_surface(A_par, B_par, Z, rstride=1, cstride=1, cmap=cm.jet)
ax.set_xlabel(' Масштаб:a')
ax.set_ylabel('Задержка: b')
ax.set_zlabel('Амплитуда ВП: $ N_{ab}$')

plt.figure('2D-график для z = w (a,b)')
plt.title('Плоскость ab с цветовыми областями ВП', size=12)
plt.contourf(B_par, A_par, Z, 100)
# plt.show()



                                                            # # # # # PyWavelet # # # # #





import numpy as np
import matplotlib.pyplot as plt

import pywt


# time, sst = pywt.data.nino()

sst=[S(t) for t in arange(0,Size,1)]
time=[t for t in arange(0,Size,1)]




# print(time)
# print(sst)
# dt = time[1] - time[0]
dt = 1



# data_size = len(time)
# print(data_size)
# Taken from http://nicolasfauchereau.github.io/climatecode/posts/wavelet-analysis-in-python/
# wavelet = 'cmor1.5-1.0'
wavelet = 'mexh'
# wavelet = 'morl'
# wavelet = 'mexh'
# wavelet = 'mexh'
max_scale = 50
scales = np.arange(1, max_scale)

[cfs, frequencies] = pywt.cwt(sst, scales, wavelet, dt)
# print(cfs)
# print(len(cfs[0]))


A_scales, B_time = meshgrid(time,scales)
# CFS = cfs.reshape(max_scale-1,data_size - 1)


plt.figure('pywt: 2D-график для z = w (a,b)')
plt.title('pywt: Плоскость ab с цветовыми областями ВП', size=12)
plt.plot(time, sst)
# plt.show()





plt.figure('pywt: 2D-график для z = w (a,b)')
plt.title('pywt: Плоскость ab с цветовыми областями ВП', size=12)
plt.contourf(A_scales, B_time, cfs, 100)
# plt.show()





power = (abs(cfs)) ** 2
period = 1. / frequencies
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
f, ax = plt.subplots(figsize=(15, 10))
ax.contourf(time, period, power, levels, extend='both')

ax.set_title("pywt Power")
ax.set_ylabel('period')













# power = (abs(cfs)) ** 2
# period = 1. / frequencies
# levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
# f, ax = plt.subplots(figsize=(15, 10))
# ax.contourf(time, np.log2(period), np.log2(power), np.log2(levels),
#             extend='both')

# ax.set_title('%s Wavelet Power Spectrum (%s)' % ('Nino1+2', wavelet))
# ax.set_ylabel('Period (years)')
# Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
#                         np.ceil(np.log2(period.max())))
# ax.set_yticks(np.log2(Yticks))
# ax.set_yticklabels(Yticks)
# ax.invert_yaxis()
# ylim = ax.get_ylim()
# ax.set_ylim(ylim[0], -1)

# plt.show()




















# A_par, B_par = meshgrid(A, B)
# Z = z.reshape(Size_A-1, Size)

# # print(Z)

# # print(len(Z))
# # print(len(Z[0]))

# fig = plt.figure('Вейвлет- спектр: гармонического колебания')
# ax = Axes3D(fig)
# # ax.plot_surface(A_par, B_par, Z, rstride=1, cstride=1, cmap=cm.jet)
# # ax.set_xlabel(' Масштаб:a')
# # ax.set_ylabel('Задержка: b')
# # ax.set_zlabel('Амплитуда ВП: $ N_{ab}$')

# plt.figure('2D-график для z = w (a,b)')
# plt.title('Плоскость ab с цветовыми областями ВП', size=12)
# plt.contourf(B_par, A_par, Z,100)




















plt.show()






