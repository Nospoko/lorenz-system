from scipy.integrate import RK45
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Define the default parameters values
    sigma = 10
    rho = 28
    beta = 8/3
    t0 = 0
    y0 = (0., 1., 1.05)
    tmax = 50
    
    def lorenz(t, xyz):
        x, y, z = xyz
        x_dot = sigma*(y - x)
        y_dot = rho*x - y - x*z
        z_dot = x*y - beta*z
        return np.array([x_dot, y_dot, z_dot])

    solution = RK45(lorenz, t0=t0, y0=y0, t_bound=tmax)
    # collect data
    xyzs = []
    it = 0
    while solution.status != 'finished' and solution.status != "failed":
        # get solution step state
        solution.step()
        xyzs.append(solution.y)
        it += 1
    
    xyzs = np.array(xyzs)
    # Plot
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(*xyzs.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    plt.show()
if __name__ == "__main__":
    main()