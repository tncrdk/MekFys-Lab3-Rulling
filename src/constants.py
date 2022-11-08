from dataclasses import dataclass


@dataclass
class Cylinder:
    name: str
    diameter: float
    mass: float
    d_diameter: float
    d_mass: float
    slope_radius: float
    # TODO Legge til phi_0

    def __post_init__(self):
        self.L = self.slope_radius - self.diameter / 2


R: float = 46
dR: float = 0.1


CYLINDERS: list[Cylinder] = [
    Cylinder(
        name="metall hul",
        diameter=0.424,
        mass=0.255,
        d_diameter=0.001,
        d_mass=0.0005,
        slope_radius=R,
        # TODO Finn phi_0
    ),
    Cylinder(
        name="plast massiv",
        diameter=0.735,
        mass=0.44,
        d_diameter=0.001,
        d_mass=0.0005,
        slope_radius=R,
    ),
    Cylinder(
        name="metall massiv",
        diameter=0.445,
        mass=1.097,
        d_diameter=0.001,
        d_mass=0.0005,
        slope_radius=R,
    ),
]
