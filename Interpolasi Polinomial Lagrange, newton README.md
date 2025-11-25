#interpolasi polinomial lagrange
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.0, 2.0, 3.0])
y = np.array([3.0, 8.0, 15.0])
xp = 2.5

def lagrange_interpolation(x, y, xp):
    p = 0.0
    n = len(x)
    for i in range(n):
        L = 1.0
        for j in range(n):
            if j != i:
              L*= (xp-x[j]) / (x[i] - x[j])
        p += y[i] * L
    return p
L_value = lagrange_interpolation(x, y, xp)
print("Lagrange f(2.5) =", L_value)

xp_val = xp
yp_val = l_value

plt.figure(figsize=(8,5))
plt.scatter(x, y, s=80, label="Titik Data", zorder=3)

x_lin = np.linspace(min(x), max(x), 200)
y_lin = [lagrange_interpolation(x, y, xx) for xx in x_lin]

plt.plot(x_lin, y_lin, linewidth=2, label="Kurva Lagrange")
plt.scatter(xp_val, yp_val, color='red', s=100, label=f"f({xp_val}) = {yp_val:.3f}")
plt.title("Grafik Interpolasi Lagrange", fontsize=14)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()

#interporasi polinomial newton
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
