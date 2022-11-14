from __future__ import annotations
from typing import Generator, Callable
from pathlib import Path
from constants import Cylinder, CYLINDERS, delta
import numpy as np
import matplotlib.pyplot as plt
from Crank_Nicholson_method import stepCN
from data_generering import ODE_solver, numerical_data_generation
from functools import partial
from scipy.stats import tstd
from scipy.optimize import least_squares

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
    return np.arcsin(
        (x_values + 0.45) / cylinder.L
    )  # Lagt til et lite skift pga skjevhet i filmen


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


def optimize_numerical():
    for path, cylinder in zip(get_filepaths(is_numerical=False), CYLINDERS):
        error_func = partial(error_numerical, filepath=path, cylinder=cylinder)
        bounds = [(0.0, 0.04), (0.0, 0.001), (0.0, 0.04)]  # delta, phi_R, beta
        guess = [0.02, 0.0001, 0.02]
        ode_model = get_ode_model(ODE_solver, cylinder)
        # guess_0 = np.array([0.01, 0.01, 0.01])  # delta, phi_R, beta
        res = least_squares(error_func, guess)

        # first = error_func(guess_0)
        # guess_0 = np.array([0.01, 0.21, 0.01])  # delta, phi_R, beta
        # second = error_func(guess_0)
        # print(first - second)
        print(res)


def get_ode_model(ODE_solver: Callable, cylinder: Cylinder):
    STEP_FUNC = stepCN
    t_0: float = 0
    t_end: float = 100
    phi_0: float = cylinder.phi_0
    phi_d1_0: float = 0
    w0: float = cylinder.w0
    dt = 0.01
    gamma = cylinder.gamma

    return partial(
        ODE_solver,
        dt=dt,
        t_0=t_0,
        t_end=t_end,
        phi_0=phi_0,
        phi_d1_0=phi_d1_0,
        w0=w0,
        gamma=gamma,
        step_func=STEP_FUNC,
    )


def error_numerical(guess: np.ndarray, filepath: Path, cylinder: Cylinder):
    # do experimental - numerical values in arrays
    # reduce std
    delta = guess[0]
    phi_R = guess[1]
    beta = guess[2]

    STEP_FUNC = stepCN
    t_0: float = 0
    t_end: float = 100
    phi_0: float = cylinder.phi_0
    phi_d1_0: float = 0
    w0: float = cylinder.w0
    dt = 0.01

    times_exp, x_values_exp = read_data(filepath)
    phi_values_exp = transform_x_to_phi(x_values_exp, cylinder)
    times_num, phi_values_num, _ = ODE_solver(
        dt,
        t_0,
        t_end,
        phi_0,
        phi_d1_0,
        w0,
        cylinder.gamma,
        phi_R=phi_R,
        delta=delta,
        beta=beta,
        step_func=STEP_FUNC,
    )

    difference = diff_numerical_experimental(
        (times_num, phi_values_num), (times_exp, phi_values_exp)
    )
    stdev = tstd(np.abs(difference))
    # s = np.sum(np.abs(difference))
    print("delta = ", delta)
    print("phi_R = ", phi_R)
    print("beta = ", beta)
    print("Stdev: ", stdev, "\n")
    # print("sum: ", s, "\n")
    # return stdev
    return difference


def compare_arr_numerical_experimental(
    num_data: tuple[np.ndarray, np.ndarray], exp_data: tuple[np.ndarray, np.ndarray]
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    times_num = np.round(num_data[0], 2)
    times_exp = np.round(exp_data[0], 2)
    values_exp = exp_data[1]
    values_num = num_data[1]

    list_indexing = [time in times_exp for time in times_num]
    times_num_filtered = times_num[list_indexing]
    num_values_filtered = values_num[list_indexing]

    list_indexing = [time in times_num_filtered for time in times_exp]
    times_exp_filtered = times_exp[list_indexing]
    values_exp = values_exp[list_indexing]

    return (times_num_filtered, num_values_filtered), (times_exp_filtered, values_exp)


def diff_numerical_experimental(
    num_data: tuple[np.ndarray, np.ndarray], exp_data: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    filtered_num, filtered_exp = compare_arr_numerical_experimental(num_data, exp_data)
    _, phi_values_num = filtered_num
    _, phi_values_exp = filtered_exp

    diff = phi_values_num - phi_values_exp
    return diff


if __name__ == "__main__":
    # numerical_data_generation()
    # generate_combined_plots("stepCN")
    # generate_numerical_plots("stepCN")
    # combine_numeric_analytic_plots("stepCN")
    optimize_numerical()
