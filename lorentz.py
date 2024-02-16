from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt


class RungeKutta4:
    def __init__(
        self,
        fun: Callable,
        t0: float,
        y0: np.ndarray | Any,
        h: float,
        num: int,
    ) -> None:
        """
        4-th order Runge Kutta ODE solver. Integration is always performed in + direction.

        Args:
            fun (Callable): Right-hand side of ODE.
            x0 (float): Beginning of integration interval
            y0 (np.ndarray | Any): Initial conditions vector
            h (float): Integration step
            num (int): Number of steps. Length of result array.
        """
        self.fun = fun
        self.h = h
        self.x0 = t0
        self.y0 = y0
        self.num = num

        # number of equations in a system
        self.n_eq = len(self.y0)

        # result and node arrays
        self.result = np.empty((num, self.n_eq))
        self.nodes = np.empty((num, self.n_eq))

        # current step
        self.it = 1
        self.result[0] = y0

        # Runge Kutta method parameters
        self.gammas = np.array([[1 / 3, 0, 0], [-1 / 3, 1, 0], [1, -1, 1]])
        self.cs = [1 / 8, 3 / 8, 3 / 8, 1 / 8]

    def step(self):
        # get previous step results
        t_i = self.nodes[self.it - 1]
        y_i = self.result[self.it - 1]

        # calculate ks
        k0 = self.h * self.fun(t_i, y_i)
        k1 = self.h * self.fun(t_i + self.h * self.gammas[0][0], y_i + self.gammas[0][0] * k0)
        k2 = self.h * self.fun(
            t_i + self.gammas[1, :].sum() * self.h,
            y_i + self.gammas[1, 0] * k1 + self.gammas[1, 1] * k1,
        )
        k3 = self.h * self.fun(
            t_i + self.gammas[2, :].sum() * self.h,
            y_i + self.gammas[2, 0] * k1 + self.gammas[2, 1] * k1 + self.gammas[2, 2] * k2,
        )

        # calculate result
        self.result[self.it] = y_i + self.cs[0] * k0 + self.cs[1] * k1 + self.cs[2] * k2 + self.cs[3] * k3
        self.it += 1
        return self.result[self.it - 1]

    def integrate(self):
        while self.it < self.num:
            self.step()
        return self.result


def main():
    # Define the default parameters values
    sigma = 10
    rho = 28
    beta = 8 / 3
    t0 = 0
    y0 = (0.0, 1.0, 1.05)
    h = 0.01
    num = 10000

    def lorenz(t, xyz):
        x, y, z = xyz
        x_dot = sigma * (y - x)
        y_dot = rho * x - y - x * z
        z_dot = x * y - beta * z
        return np.array([x_dot, y_dot, z_dot])

    rk4 = RungeKutta4(lorenz, t0=t0, y0=y0, h=h, num=num)
    xyzs = rk4.integrate()

    # Plot
    ax = plt.figure().add_subplot(projection="3d")

    ax.plot(*xyzs.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    plt.show()


if __name__ == "__main__":
    main()
