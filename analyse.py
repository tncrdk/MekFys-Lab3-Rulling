from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import Crank_Nicholson_method as CN


def read_data(filename: str) -> tuple[np.ndarray]:
    with open(filename, "r") as f:
        pass
        # TODO Format data


def plot_results(
    filename: str,
    x_values_list: list[np.ndarray],
    y_values_list: list[np.ndarray],
    labels: list[str],
    xlabel: str,
    ylabel: str,
) -> None:
    for x_values, y_values, label in zip(x_values_list, y_values_list, labels):
        plt.plot(x_values, y_values, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(fontsize=18)
    plt.grid()

    plt.savefig(filename, bbox_inches="tight")


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
    


def ODE_solver(
    dt: float,
    t_0: float,
    t_end: float,
    phi_0: float,
    phi_dot_0: float,
    w0: float,
    gamma: float = 1,
    phiR: float = 0.0,
    delta: float = 0.0,
    beta: float = 0.0,
    step_func: Callable = step_euler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pass


if __name__ == "__main__":
    pass
