import numpy as np
import matplotlib.pyplot as plt

x = np.array([10, 20, 30, 40])
y = np.array([2.5, 4, 5, 6.5])

n = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)

b1 =np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) **2)
b0 = y_mean - b1 * x_mean

print("Persamaan regresi: y = {:.4f}x + {:.4f}".format (b1, b0))

x_prediction = 100
y_prediction = b0 + b1 * x_prediction
print("Prediction waktu muat ukuran 25 MB =", y_prediction)

x_line = np.linspace(min(x), max(x), 100)
y_line = b0 + b1 * x_line

plt.scatter(x, y, label="Data Asli", s=70)
plt.plot(x_line, y_line, label="Garis Regresi Linier")
plt.scatter(x_prediction, y_prediction, color="red", label=f"Prediksi x={x_prediction}")

plt.xlabel("Ukuran File (MB)")
plt.ylabel("Waktu Muat (detik)")
plt.title("Grafik Regresi Linier")
plt.legend()
plt.grid(True)
plt.show()
