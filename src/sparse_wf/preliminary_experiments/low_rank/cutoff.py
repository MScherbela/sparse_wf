# %%
import jax
import matplotlib.pyplot as plt
import numpy as np


def cutoff(d, p=4):
    a = -(p + 1) * (p + 2) * 0.5
    b = p * (p + 2)
    c = -p * (p + 1) * 0.5
    return 1 + a * d**p + b * d ** (p + 1) + c * d ** (p + 2)


if __name__ == "__main__":
    plt.close("all")
    x = np.linspace(0, 1, 100)

    evaluations = []
    n_derivs = 3
    p = 4
    func = cutoff
    for n in range(n_derivs):
        f_vmapped = jax.vmap(func, in_axes=(0, None))
        evaluations.append(f_vmapped(x, p))
        func = jax.grad(func)

    for n, y in enumerate(evaluations):
        plt.plot(x, y, label=f"$f^{{({n})}}$")
    plt.legend()
