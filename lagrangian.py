import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import gaussian_filter


############################################################
                # Parâmetros
############################################################

c0 = 1500
b = 2
z0 = 10**3
x0 = 0
za = 1300
eps = 0.00737
dl = 10
L = 50*(10**3)
maxz = 5*(10**3)
zz = np.arange(0, maxz, 0.1)
dtheta = 0.005
thetaInterval = np.arange(-0.1, 0.1, dtheta)
N = int(L / dl)

############################################################
                # Funções 
############################################################


def c(z):
    zbar = b * (z - za) / za
    return c0 * (1.0 + eps*(zbar - 1.0 + np.exp(-zbar)))

def dc(z):
    zbar = b * (z - za) / za
    return (c0*eps*b/za)*(1.0 - np.exp(-zbar))

def rk2(x, z, xi, eta, dl):
    cc = c(z)
    dcc = dc(z)

    x1 = x + dl*cc*xi
    z1 = z + dl*cc*eta
    xi1 = xi
    eta1 = eta - dl*dcc/(cc**2)

    c2 = c(z1)
    dc2 = dc(z1)

    x2 = x + dl * c2 * xi
    z2 = z + dl * c2 * eta
    xi2 = xi
    eta2 = eta - dl * dc2 / (c2**2)

    return (x1 + x2)/2, (z1 + z2)/2, (xi1 + xi2)/2, (eta1 + eta2)/2


def createRays(angles):
    rays = []
    for ang in angles:
        x = np.array([])
        z = np.array([])
        xi = np.array([])
        eta = np.array([])
        S = np.array([])

        x = np.append(x, x0)
        z = np.append(z, z0)
        S = np.append(S, 0)

        c1 = c(z[0])
        
        xi = np.append(xi, np.cos(ang)/c1)
        eta = np.append(eta, np.sin(ang)/c1)

        for i in range(N):
            xn, zn, xin, etan = rk2(x[i], z[i], xi[i], eta[i], dl)
            Sn = S[i] + dl / c(z[i])
            x = np.append(x, xn)
            z = np.append(z, zn)
            xi = np.append(xi, xin)
            eta = np.append(eta, etan)
            S = np.append(S, Sn)

        rays.append((x, z))
        
    return rays

def plotRays(rays):
    plt.figure(figsize=(8,5))
    for ray in rays:
        plt.plot(ray["x"], ray["z"])

    plt.gca().invert_yaxis()
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("Ray tracing")
    plt.grid(True)
    plt.show()


def evaluateA0(rays):
    A0 = []

    for k in range(1, len(rays) - 1):
        xm, zm = rays[k-1]
        xp, zp = rays[k+1]
        xk, zk = rays[k]

        dx_dth = (xp - xm) / (2 * dtheta)
        dz_dth = (zp - zm) / (2 * dtheta)

        J = np.sqrt(dx_dth**2 + dz_dth**2)
        J[J < 1e-6] = 1e-6  # regularização

        A = np.sqrt(c(zk) / c(zk[0])) / np.sqrt(J)
        A0.append(A)

    return A0

def evaluateSolution(A0):
    nx, nz = 500, 300
    xg = np.linspace(0, 50_000, nx)
    zg = np.linspace(500, 2500, nz)

    dx = xg[1] - xg[0]
    dz = zg[1] - zg[0]

    I = np.zeros((nz, nx))

    for k in range(len(A0)):
        xk, zk = rays[k+1]
        Ak = A0[k]

        for i in range(len(xk)):
            ix = int((xk[i] - xg[0]) / dx)
            iz = int((zk[i] - zg[0]) / dz)

            if 0 <= ix < nx and 0 <= iz < nz:
                I[iz, ix] += Ak[i]**2

    I = gaussian_filter(I, sigma=2)
    return xg, zg, I

def plotSolution(xg, zg, I):
    V = 10 * np.log10(I + 1e-12)

    plt.figure(figsize=(10,5))
    levels = np.linspace(-40, 0, 60)

    plt.contourf(xg, zg, V, levels=levels, cmap="jet", extend="both")
    plt.colorbar(label="Intensity (dB)")
    plt.gca().invert_yaxis()
    plt.xlabel("Range (m)")
    plt.ylabel("Depth (m)")
    plt.title("u(x,y)²")
    plt.show()


############################################################
                # Aplicando 
############################################################

rays = createRays(thetaInterval)
# plotRays(rays)
A0 = evaluateA0(rays)
xg, zg, I = evaluateSolution(A0)
plotSolution(xg, zg, I)