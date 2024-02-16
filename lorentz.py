import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from solvers import RungeKutta4


def main():
    parameter_cols = st.columns(3)
    # Define the default parameters values
    with parameter_cols[0]:
        sigma = st.number_input(label="sigma", value=10.0)
        y_1 = st.number_input(label="y_1", value=0.0)
        x0 = st.number_input(label="x0", value=0.0)
    with parameter_cols[1]:
        rho = st.number_input(label="rho", value=28.0)
        y_2 = st.number_input(label="y_2", value=1.0)
        h = st.number_input(label="step size", value=0.01)
    with parameter_cols[2]:
        beta = st.number_input(label="beta", value=8 / 3)
        y_3 = st.number_input(label="y_3", value=1.05)
        num = st.number_input(label="number of steps", value=10000, step=100)

    y0 = (y_1, y_2, y_3)

    def lorenz(t, y):
        y_1, y_2, y_3 = y
        y1_dot = sigma * (y_2 - y_1)
        y2_dot = rho * y_1 - y_2 - y_1 * y_3
        y3_dot = y_1 * y_2 - beta * y_3
        return np.array([y1_dot, y2_dot, y3_dot])

    rk4 = RungeKutta4(lorenz, x0=x0, y0=y0, h=h, num=num)
    result = rk4.integrate()

    fig, axes = plt.subplots(3, figsize=(14, 7))
    for it, ax in enumerate(axes):
        ax.plot(rk4.nodes, result.T[it], label=f"y_{it + 1}")
        ax.legend()
    axes[0].set_title("Y(x)")
    st.pyplot(fig)

    # Plot
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(projection="3d")

    ax.plot(*result.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    st.pyplot(fig)


if __name__ == "__main__":
    main()
