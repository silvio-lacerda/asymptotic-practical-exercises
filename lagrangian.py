import numpy as np
import matplotlib.pyplot as plt


c0 = 1500
b = 2
za = 1300
eps = 0.00737
dl = 10
L = 50*(10**3)
maxz = 5*(10**3)
zz = np.arange(0, maxz, 0.1)


N = int(L / dl)

def c(z):
    zbar = b * (z - za) / za
    return c0 * (1.0 + eps*(zbar - 1.0 + np.exp(-zbar)))

def dcdz(z):
    zbar = b * (z - za) / za
    return (c0*eps*b/za)*(1.0 - np.exp(-zbar))

def rk_step(x, z, xi, eta, dl):
    cc = c(z)
    dc = dcdz(z)

    x1 = x + dl*cc*xi
    z1 = z + dl*cc*eta
    xi1 = xi
    eta1 = eta - dl*dc/(cc**2)

    c2 = c(z1)
    dc2 = dcdz(z1)

    x2 = x + dl * c2 * xi
    z2 = z + dl * c2 * eta
    xi2 = xi
    eta2 = eta - dl * dc2 / (c2**2)

    return (x1 + x2)/2, (z1 + z2)/2, (xi1 + xi2)/2, (eta1 + eta2)/2

plt.figure(figsize=(8,5))

dtheta = 0.005
angInterval = np.arange(-0.1, 0.1001, dtheta)
cc = c(zz)

rays = []

for ang in angInterval:

    x = np.zeros(N + 1)
    z = np.zeros(N + 1)
    xi = np.zeros(N + 1)
    eta = np.zeros(N + 1)
    S = np.zeros(N + 1)

    x[0] = 0.0
    z[0] = 1000.0

    c1 = c(z[0])

    xi[0] = np.cos(ang) / c1
    eta[0] = np.sin(ang) / c1

    for i in range(N):
        x[i+1], z[i+1], xi[i+1], eta[i+1] = rk_step(x[i], z[i], xi[i], eta[i], dl)
        S[i+1] = S[i] + dl / c(z[i])

    rays.append({"x": x, "z": z, "S": S})

for ray in rays:
    plt.plot(ray["x"], ray["z"])

plt.gca().invert_yaxis()
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title("Ray tracing")
plt.grid(True)
plt.show()


# A0 = []

# for k in range(1, len(rays)-1):
#     xm = rays[k-1]["x"]
#     zm = rays[k-1]["z"]
#     xp = rays[k+1]["x"]
#     zp = rays[k+1]["z"]

#     x0 = rays[k]["x"]
#     z0 = rays[k]["z"]

#     dxdth = (xp - xm) / (2*dtheta)
#     dzdth = (zp - zm) / (2*dtheta)

#     J = np.sqrt(dxdth**2 + dzdth**2)

#     # estou usando que l_0 = 1, A_0(l_0) = 1

#     A = np.sqrt(c(z0) / c(z0[0])) / np.sqrt(J)
#     A0.append(A)



# mid = len(A0)//2

# # 1j é como o python entende a unidade imaginária
# u = A0[mid] * np.exp(1j * rays[mid+1]["S"])

# plt.figure()

# plt.plot(rays[mid+1]["x"], np.abs(u))
# plt.xlabel("x (m)")
# plt.ylabel("|u|")
# plt.title("Amplitude |u| ao longo do raio central")
# plt.grid(True)

# plt.show()