from dataclasses import dataclass
import numpy as np


R: float = 46
dR: float = 0.1
g = 981
dt = 0.01

delta, phi_R, beta = (0.014, 0.0002, 0.080)


@dataclass
class Cylinder:
    name: str
    diameter: float
    mass: float
    d_diameter: float
    d_mass: float
    slope_radius: float
    x_0: float
    c: float

    def __post_init__(self):
        self.L = self.slope_radius - self.diameter / 2
        self.phi_0 = np.arcsin(self.x_0 / self.L)
        self.gamma = 1 / (1 + self.c)
        self.w0 = np.sqrt(self.gamma * g / self.L)


CYLINDERS: list[Cylinder] = [
    Cylinder(
        name="hul_metall",
        diameter=0.424,
        mass=0.255,
        d_diameter=0.001,
        d_mass=0.0005,
        slope_radius=R,
        x_0=5.910406,
        c=0.5,
    ),
    Cylinder(
        name="massiv_metall",
        diameter=0.445,
        mass=1.097,
        d_diameter=0.001,
        d_mass=0.0005,
        slope_radius=R,
        x_0=-1.023518e1,
        c=0.5,
    ),
    Cylinder(
        name="massiv_plast",
        diameter=0.735,
        mass=0.44,
        d_diameter=0.001,
        d_mass=0.0005,
        slope_radius=R,
        x_0=-1.028708e1,
        c=0.5,
    ),
]
