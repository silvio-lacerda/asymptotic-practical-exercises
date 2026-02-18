import numpy as np
import matplotlib.pyplot as plt


c = 1500
H = 50
zs = 10
zr = 25
xr = 10000
sigma = 0.005
jmax = 20
tmin = xr/c + 1         # antes de xr/c dá problema nas raízes de k e fica gerando valores nulos
tmax = xr*50/c
t = np.linspace(tmin, tmax, 800)
u = np.zeros_like(t)

def sign(a):
    if a > 0:
        return 1
    elif a == 0:
        return 0
    else:
        return -1
 
def asymptotic(h,A,Hess,Phi):
    return np.sqrt(2*np.pi*h)*np.exp(1j*np.pi*sign(Hess)/4)*A*np.exp(1j*Phi/h)/np.sqrt(np.abs(Hess))

def phi(j,z):
    return np.sqrt(2/H)*np.sin(j*np.pi*z/H)

def k(j,w):
    A = (w/c)**2
    B = (j*np.pi/H)**2
    return np.sqrt(A - B)

def FS(w):
    return np.exp(-(w**2)*((sigma**2)/2))

def stationaryPoint(tt,j):
    A = j*np.pi/H
    B = np.sqrt(((c*tt)**2)-(xr**2))
    return ((c**2)*tt*A)/B

def der2Phi(j,w):
    B = j*np.pi/H
    kc = k(j, w)
    return -xr*(B**2)/((c**2)*(kc**3))

def A(j,w):
    return (1j/2)*phi(j,zr)*phi(j,zs)*FS(w)/k(j,w)

def Phi(j,w,tt):
    return k(j,w)*xr - w*tt

def run():
    for j in range(1, jmax + 1):
        for it, tt in enumerate(t):
            wsp = stationaryPoint(tt,j) 
            h = 1/xr
            amp = A(j,wsp)
            Hess = der2Phi(j,wsp)
            phase = Phi(j,wsp,tt)
            u[it] += asymptotic(h,amp,Hess,phase).real


def plotSolution():
    plt.figure(figsize=(10, 5))
    plt.plot(t, u, lw=2)
    plt.xlabel("Time")
    plt.ylabel("Solution u(t, xr, zr)")
    plt.title("Stationary Phase Method")
    plt.grid(True)
    plt.show()


run()
plotSolution()