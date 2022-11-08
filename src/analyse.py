from __future__ import annotations
from typing import Generator
from pathlib import Path
from constants import Cylinder, CYLINDERS, delta
import numpy as np
import matplotlib.pyplot as plt


def get_filepaths(
    is_numerical: bool, step_func_name: str = "step-euler"
) -> Generator[Path, None, None]:
    workspace_root = Path(__file__).parent.parent
    if is_numerical:
        data_files_folder = workspace_root / "Data" / "Numerical" / step_func_name
    else:
        data_files_folder = workspace_root / "Data" / "Experimental"
    data_file_paths = Path(data_files_folder).glob("*.txt")
    return data_file_paths


def read_data(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    values: list[float] = []
    with open(filepath, "r") as f:
        f.readline()
        f.readline()
        for line in f.readlines():
            line_list = line.split()
            times.append(float(line_list[0]))
            values.append(float(line_list[1]))
        return np.array(times), np.array(values)


def transform_x_to_phi(x_values: np.ndarray, cylinder: Cylinder) -> np.ndarray:
    return np.arcsin(x_values / cylinder.L)


def analytic_solution(t, phi_0, w0, delta):
    wd = np.sqrt(w0**2 - delta**2)
    return phi_0 * np.exp(-delta * t) * np.cos(wd * t)


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

    plt.legend()
    plt.grid()

    plt.savefig(filename, bbox_inches="tight")
    plt.clf()


def combine_numeric_analytic_plots(step_func_name: str):
    numerical_filepaths = get_filepaths(True, step_func_name)

    for num_path, cylinder in zip(numerical_filepaths, CYLINDERS):
        times_num, phi_values_num = read_data(num_path)
        x_values_analytic = np.linspace(times_num[0], times_num[-1], len(times_num))
        phi_values_analytic = analytic_solution(
            x_values_analytic, cylinder.phi_0, cylinder.w0, delta
        )

        plot_path = (
            Path(__file__).parent.parent
            / "Plots"
            / "Combined"
            / f"{num_path.name.replace('-data.txt', '')}.png"
        )
        num_label = f"{num_path.name.replace('-data.txt', '')}"
        analytic_label = f"Analytic solution"

        plot_results(
            plot_path,
            [times_num, times_num],
            [phi_values_num, phi_values_analytic],
            [num_label, analytic_label],
            "tid (s)",
            "vinkelutslag (rad)",
        )


def generate_experimental_plots():
    file_paths = get_filepaths(False)
    for path, cylinder in zip(file_paths, CYLINDERS):
        times, x_values = read_data(path)
        phi_values = transform_x_to_phi(x_values * 100, cylinder)
        plot_path = (
            Path(__file__).parent.parent
            / "Plots"
            / "Experimental"
            / f"{cylinder.name}-plot"
        )
        plot_results(
            plot_path,
            [times],
            [phi_values],
            [f"{cylinder.name}"],
            "tid (s)",
            "vinkelutslag (rad)",
        )


def generate_numerical_plots(step_func_name: str = "step_euler"):
    file_paths = get_filepaths(True, step_func_name=step_func_name)
    for path in file_paths:
        times, phi_values = read_data(path)
        plot_path = (
            Path(__file__).parent.parent
            / "Plots"
            / "Numerical"
            / step_func_name
            / (path.name.replace("data.txt", "plot") + ".png")
        )
        label = f"{path.name.replace('-data.txt', '')}"
        plot_results(
            plot_path,
            [times],
            [phi_values],
            [label],
            xlabel="tid (s)",
            ylabel="vinkelutslag (rad)",
        )


def generate_combined_plots(step_func_name: str):
    numerical_filepaths = get_filepaths(True, step_func_name)
    experimental_filepaths = get_filepaths(False)
    for num_path, exp_path, cylinder in zip(
        numerical_filepaths, experimental_filepaths, CYLINDERS
    ):
        times_num, phi_values_num = read_data(num_path)
        times_exp, x_values_exp = read_data(exp_path)
        phi_values_exp = transform_x_to_phi(x_values_exp * 100, cylinder)

        max_time = min([max(times_num), max(times_exp)])
        times_num_filtered = times_num[times_num < max_time]
        times_exp_filtered = times_exp[times_exp < max_time]
        phi_values_num_filtered = phi_values_num[: len(times_num_filtered)]
        phi_values_exp_filtered = phi_values_exp[: len(times_exp_filtered)]

        plot_path = (
            Path(__file__).parent.parent
            / "Plots"
            / "Combined"
            / f"{num_path.name.replace('-data.txt', '')}.png"
        )
        num_label = f"{num_path.name.replace('-data.txt', '')}"
        exp_label = f"{exp_path.name.replace('_data.txt', '-exp')}"

        plot_results(
            plot_path,
            [times_num_filtered, times_exp_filtered],
            [phi_values_num_filtered, phi_values_exp_filtered],
            [num_label, exp_label],
            "tid (s)",
            "vinkelutslag (rad)",
        )


if __name__ == "__main__":
    generate_combined_plots("stepCN")
    # generate_numerical_plots("stepCN")
    # combine_numeric_analytic_plots("stepCN")
