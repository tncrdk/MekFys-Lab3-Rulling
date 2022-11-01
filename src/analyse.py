from typing import Callable, Iterator
import numpy as np
import matplotlib.pyplot as plt
import Crank_Nicholson_method as CN
from pathlib import Path


def main():
    file_paths = get_filepaths()
    return read_data(next(file_paths))


def get_filepaths() -> Iterator[Path]:
    workspace_root = Path(__file__).parent.parent
    data_files_folder = workspace_root / "data"
    file_paths = Path(data_files_folder).glob("*.txt")
    return file_paths


def read_data(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[int] = []
    x_values: list[int] = []
    with open(filepath, "r") as f:
        f.readline()
        f.readline()
        for line in f.readline():
            line_list = line.split()
            times.append(int(line_list[0]))
            x_values.append(int(line_list[1]))
        return np.array(times), np.array(x_values)


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
    new_phi = phi + phi_d1 * dt
    new_phi_d1 = phi_d1 + phi_d2(phi, phi_d1, w0, delta, beta, phi_R, gamma)
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


if __name__ == "__main__":
    print(main())
