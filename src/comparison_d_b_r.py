# from scipy import constants as c
import scipy.constants as c
from numpy import sqrt, cos, linspace, pi, gradient
import matplotlib.pyplot as np
from pathlib import Path

# CONST
g = 9.81
m = 2       # kg
r = 0.1     # m
l = 0.46-r

c = 0.5*m*r**2
gamma = 1/(1+c)
omega = sqrt(gamma*g/l)

# MÅLTE
delta = 0.014
phi_R = 0.0002
beta = 0.080
# delta = 0.02
# phi_R = 0.02
# beta = 0.02

def F_s(phi, phi_d):
    return 2*delta*phi_d

def F_r(phi, phi_d):
    return pi*phi_R/(2*omega)*(omega**2*cos(phi) + gamma*phi_d**2)

def F_d(phi, phi_d):
    return beta*(3*pi)/(4*omega)*phi_d**2

# phis = linspace(0, pi/8, 100)
phis = pi/4
phi_ds = linspace(0, 0.83, 100)

np.figure()
np.plot(phi_ds, F_s(phis, phi_ds), label = "Dempingskraft-størrelse")
np.plot(phi_ds, F_r(phis, phi_ds), label= "Rullefriksjon-størrelse")
np.plot(phi_ds, F_d(phis, phi_ds), label= "Luftmotstand-størrelse")
# np.plot(phi_ds, gradient(F_r(phis, phi_ds)), label="grad")
np.xlabel("vinkelfart (rad/s)")
np.ylabel("demping")
np.legend()
# np.show()
np.savefig(Path(__file__).parent.parent / "Plots" / "comparison_d_b_r.png")

# print(gradient(F_rs(phis, phi_ds)))