import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([1.0, 2.0, 3.0])
y = np.array([3.0, 8.0, 15.0])
xp = 2.5

def newton_interpolation(x, y, xp):
    n = len(x)
    diff = np.zeros((n, n))
    diff[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            diff[i][j] = (diff[i+1][j-1] - diff[i][j-1]) / (x[i+j] - x[i])

    result = diff[0][0]
    product = 1.0

    for j in range(1, n):
        product *= (xp - x[j-1])
        result += diff[0][j] * product

    return result, diff


N, diff_table = newton_interpolation(x, y, xp)
print("Hasil Newton f(2.5) =", N)

x_plot = np.linspace(min(x), max(x), 200)
y_plot = []
for xv in x_plot:
    yv, _ = newton_interpolation(x, y, xv)
    y_plot.append(yv)

plt.plot(x_plot, y_plot, label="Newton Interpolation")
plt.scatter(x, y, color="red", label="Data Titik")
plt.scatter(xp, N, color="green", label=f"f({xp})")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Grafik Newton Interpolation")
plt.legend()
plt.grid(True)
plt.show()
