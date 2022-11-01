import numpy as np

""" 
NB! Denne kodeblokken inneholder kode dere trenger for å kjøre Crank-Nicholson-metoden.
Det er ikke forventet at dere skal sette dere inn i hva som skjer i denne koden, men dere må klare å bruke den.
"""


def derivative(x, w0, gamma, phiR, delta, beta):
    return np.array(
        [
            x[1],
            -(w0**2) * np.sin(x[0])
            - (phiR * np.pi / (2 * w0))
            * np.sign(x[1])
            * (w0**2 * np.cos(x[0]) + gamma * x[1] ** 2)
            - 2 * delta * x[1]
            - beta * (3 * np.pi / (4 * w0)) * x[1] ** 2 * np.sign(x[1]),
        ]
    )


def findInvJac(x, w0, gamma, phiR, delta, beta, dt):
    f00 = 1
    f01 = -(dt / 2)
    f10 = -(dt / 2) * (
        -(w0**2) * np.cos(x[0])
        + (phiR * np.pi / (2 * w0)) * np.sign(x[1]) * w0**2 * np.sin(x[0])
    )
    f11 = 1 - (dt / 2) * (
        -(phiR * np.pi / (2 * w0)) * np.sign(x[1]) * 2 * gamma * x[1]
        - 2 * delta
        - 2 * beta * (3 * np.pi / (4 * w0)) * np.absolute(x[1])
    )
    return (1 / (f00 * f11 - f01 * f10)) * np.array([[f11, -f01], [-f10, f00]])


def findf(x, xPrev, w0, gamma, phiR, delta, beta, dt):
    return (
        x
        - xPrev
        - (dt / 2)
        * (
            derivative(x, w0, gamma, phiR, delta, beta)
            + derivative(xPrev, w0, gamma, phiR, delta, beta)
        )
    )


def NewtonMethod(x, xPrev, w0, gamma, phiR, delta, beta, dt):
    return x - np.matmul(
        findInvJac(x, w0, gamma, phiR, delta, beta, dt),
        findf(x, xPrev, w0, gamma, phiR, delta, beta, dt),
    )


def careFullStep(xPrev, w0, gamma, phiR, delta, beta, dt, n):
    nIntervalls = n
    dt = dt / nIntervalls

    k = np.empty((4, 2))
    x = np.empty((2, 2))
    x[0] = np.copy(xPrev)
    for i in range(nIntervalls):
        k[0] = derivative(x[0], w0, gamma, phiR, delta, beta)
        x[1] = x[0] + (dt / 2) * k[0]
        k[1] = derivative(x[1], w0, gamma, phiR, delta, beta)
        x[1] = x[0] + (dt / 2) * k[1]
        k[2] = derivative(x[1], w0, gamma, phiR, delta, beta)
        x[1] = x[0] + dt * k[2]
        k[3] = derivative(x[1], w0, gamma, phiR, delta, beta)
        x[0] += (dt / 6) * (k[0] + 2 * k[1] + 2 * k[2] + k[3])
    return x[0]


def stepCN(phi, phidot, dt, w0, gamma=1.0, phiR=0.0, delta=0.0, beta=0.0):
    """
    :param phi: float
        The previous value of phi
    :param phidot: float
        The previous value of phidot
    :param dt: float
        Fixed step size
    :param w0: float
        Undamped resonance-frequency of the harmonic oscillator
    :param gamma: float
        Factor incorporating the geometry of the oscillating object, defaults to 1
    :param phiR: float
        Zeroth order in phidot damping strength, defaults to 0
    :param delta: float
        First order in phidot damping strength, defaults to 0
    :param beta: float
        Second order in phidot damping strength, defaults to 0
    :return x[0]: float
        The next value of phi after dt time has passed
    :return x[1]: float
        The next value of phidot after dt time has passed
    :return counter: int
        NB! This variable is commented out, but can be reintroduced if you need help debugging your code.
        The number of iterations of Newtons method. If counter=-1 the derivative is non-continuous on the interval
        or if counter=-2 The Newtons-method is non-converging, and several iterations of a fourth-order RK4 has
        been employed instead
    """
    xPrev = np.array([phi, phidot])
    x = xPrev + dt * derivative(xPrev, w0, gamma, phiR, delta, beta)
    counter = 0
    tol = 1e-15  # Tolerance of Newtons method, default value 1e-15
    while (
        np.amax(np.absolute((findf(x, xPrev, w0, gamma, phiR, delta, beta, dt)))) > tol
        or counter < 1
    ):
        if (np.sign(x[1] * xPrev[1]) == -1) and counter > 2:
            return careFullStep(xPrev, w0, gamma, phiR, delta, beta, dt, 10)  # , -1
        x = NewtonMethod(x, xPrev, w0, gamma, phiR, delta, beta, dt)
        counter += 1
        if counter >= 10:
            return careFullStep(xPrev, w0, gamma, phiR, delta, beta, dt, 100)  # , -2
    return x[0], x[1]  # , counter
