from sklearn.linear_model import LinearRegression
import numpy as np

# Imagine 5 days of temperatures (°C)
# `x` is the input feature (temperature), reshaped to a column vector
x = np.array([25, 27, 30, 32, 35]).reshape(-1, 1)
# `y` is the output label (humidity percentage)
y = np.array([50, 55, 63, 70, 74])

model = LinearRegression()
model.fit(x, y)

pred = model.predict([[28]])
print(f"Predicted humidity for 28°C: {pred[0]:.2f}%")

import matplotlib.pyplot as plt

plt.scatter(x, y, color='blue', label='data')
plt.plot(x, model.predict(x), color='red', label='model')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.legend()
plt.tight_layout()
plt.savefig("results/intro_regression.png")
print("✅ Saved results/intro_regression.png")

print("slope:", model.coef_)
print("intercept:", model.intercept_)