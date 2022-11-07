from __future__ import annotations
from typing import Generator
from pathlib import Path
from constants import Cylinder, CYLINDERS
import numpy as np
import matplotlib.pyplot as plt


def get_filepaths() -> Generator[Path, None, None]:
    workspace_root = Path(__file__).parent.parent
    data_files_folder = workspace_root / "data"
    data_file_paths = Path(data_files_folder).glob("*.txt")
    return data_file_paths


def read_data(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    x_values: list[float] = []
    with open(filepath, "r") as f:
        f.readline()
        f.readline()
        for line in f.readlines():
            line_list = line.split()
            times.append(float(line_list[0]))
            x_values.append(float(line_list[1]))
        return np.array(times), np.array(x_values) * 100


def transform_x_to_phi(x_values: np.ndarray, cylinder: Cylinder) -> np.ndarray:
    return np.arcsin(x_values / cylinder.L)


def plot_results(
    filename: Path,
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
    plt.clf()


def analyze_experimental_data():
    file_paths = get_filepaths()
    for path, cylinder in zip(file_paths, CYLINDERS):
        times, x_values = read_data(path)
        phi_values = transform_x_to_phi(x_values, cylinder)
        plot_path = Path(__file__).parent.parent / "Plots" / f"{cylinder.name}-plot"
        plot_results(
            plot_path,
            [times],
            [phi_values],
            [f"{cylinder.name}"],
            "tid (s)",
            "vinkelutslag (rad)",
        )


if __name__ == "__main__":
    analyze_experimental_data()
