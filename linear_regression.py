import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Simulating house size vs price data
np.random.seed(42)
house_size = np.random.randint(500, 3500, 100).reshape(-1, 1)
noise = np.random.randn(100, 1) * 20000
house_price = (house_size * 150) + 50000 + noise

X_train, X_test, y_train, y_test = train_test_split(
    house_size, house_price, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Slope: {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:,.2f}")
print(f"R² Score: {r2:.4f}")

#Visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price (₹)")
plt.title("Linear Regression Fit")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")

plt.tight_layout()
plt.savefig("regression_result.png")
print("Chart saved as regression_result.png")