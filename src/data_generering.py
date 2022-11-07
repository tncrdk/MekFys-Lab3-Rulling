import Crank_Nicholson_method as CN
import numpy as np
from typing import Callable
from pathlib import Path
from constants import CYLINDERS


def save_numerical_results(
    filepath: Path, data: tuple[np.ndarray, np.ndarray, np.ndarray]
):
    times, phi_values, phi_d1_values = data
    with open(filepath, "w") as file:
        file.write("times phi_values phi_d1_values\n")

    with open(filepath, "a") as file:
        for t, p, p_d1 in zip(times, phi_values, phi_d1_values):
            file.write(f"{t} {p} {p_d1}\n")


def phi_d2(
    phi: float,
    phi_d1: float,
    w0: float,
    delta: float,
    beta: float,
    phi_R: float,
    gamma: float,
) -> float:
    return (
        -(w0**2) * np.sin(phi)
        - 2 * delta * phi_d1
        - np.pi
        * phi
        * phi_R
        * (w0**2 * np.cos(phi) + gamma * phi_d1**2)
        * np.sign(phi_d1)
        / (2 * w0)
        - beta * 3 * np.pi * phi_d1**2 * np.sign(phi_d1) / (4 * w0)
    )


def step_euler(
    phi: float,
    phi_d1: float,
    dt: float,
    w0: float,
    gamma: float = 1.0,
    phi_R: float = 0.0,
    delta: float = 0.0,
    beta: float = 0.0,
) -> tuple[float, float]:
    new_phi = phi + phi_d1 * dt
    new_phi_d1 = phi_d1 + phi_d2(phi, phi_d1, w0, delta, beta, phi_R, gamma) * dt
    return new_phi, new_phi_d1


def ODE_solver(
    dt: float,
    t_0: float,
    t_end: float,
    phi_0: float,
    phi_d1_0: float,
    w0: float,
    gamma: float = 1,
    phi_R: float = 0.0,
    delta: float = 0.0,
    beta: float = 0.0,
    step_func: Callable = step_euler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_datapoints = int((t_end - t_0) / dt)
    times = np.linspace(t_0, t_end, num_datapoints)
    phi_values = np.zeros(num_datapoints)
    phi_d1_values = np.zeros(num_datapoints)

    phi_values[0] = phi_0
    phi_d1_values[0] = phi_d1_0

    for i in range(1, num_datapoints):
        phi_values[i], phi_d1_values[i] = step_func(
            phi_values[i - 1], phi_d1_values[i - 1], dt, w0, gamma, phi_R, delta, beta
        )

    return times, phi_values, phi_d1_values


def numerical_data_generation():
    STEP_FUNC = step_euler
    dt = 0.1
    t_0: float = 0
    t_end: float = 100
    for cylinder in CYLINDERS:
        phi_0: float  # TODO Finn ut phi_0, og w0
        phi_d1_0: float = 0
        w0: float
        data = ODE_solver(dt, t_0, t_end, phi_0, phi_d1_0, w0, step_func=STEP_FUNC)
        data_path = (
            Path(__file__).parent.parent
            / "Data"
            / "Numerical"
            / f"{cylinder.name}-numerical-data.txt"
        )
        save_numerical_results(data_path, data)
