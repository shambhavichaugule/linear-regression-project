from dotenv import load_dotenv
import os
from huggingface_hub import login
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datasets import load_dataset
import pandas as pd

# Create/Load Dataset
# # Simulating house size vs price data
# np.random.seed(42)
# house_size = np.random.randint(500, 3500, 100).reshape(-1, 1)
# noise = np.random.randn(100, 1) * 20000
# house_price = (house_size * 150) + 50000 + noise

dataset = load_dataset("jayaprakash-m/linearRegressionDS")
df = dataset['train'].to_pandas()
df_clean = df[['Year', 'Kilometer', 'Engine', 'Max Power', 'Max Torque', 'Length', 'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity', 'Price']]

# Convert columns to numeric
df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
df_clean['Kilometer'] = pd.to_numeric(df_clean['Kilometer'], errors='coerce')
df_clean['Engine'] = pd.to_numeric(df_clean['Engine'].astype(str).str.replace(' cc', ''), errors='coerce')
df_clean['Max Power'] = pd.to_numeric(df_clean['Max Power'].astype(str).str.replace(' bhp', ''), errors='coerce')
df_clean['Max Torque'] = pd.to_numeric(df_clean['Max Torque'].astype(str).str.replace(' Nm', ''), errors='coerce')
df_clean['Length'] = pd.to_numeric(df_clean['Length'], errors='coerce')
df_clean['Width'] = pd.to_numeric(df_clean['Width'], errors='coerce')
df_clean['Height'] = pd.to_numeric(df_clean['Height'], errors='coerce')
df_clean['Seating Capacity'] = pd.to_numeric(df_clean['Seating Capacity'], errors='coerce')
df_clean['Fuel Tank Capacity'] = pd.to_numeric(df_clean['Fuel Tank Capacity'], errors='coerce')

# Fill NaN with column means
df_clean.fillna(df_clean.mean(), inplace=True)

# Drop columns that are still all NaN (if any)
df_clean = df_clean.dropna(axis=1, how='all')

print("df_clean columns:", df_clean.columns.tolist())
print("df_clean dtypes:\n", df_clean.dtypes)

X = df_clean.drop('Price', axis=1).values        
Y = df_clean['Price'].values 

features = list(df_clean.drop('Price', axis=1).columns)

print("Sample X:", X[:5])
print("Sample Y:", Y[:5])
   
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

features = list(df_clean.drop('Price', axis=1).columns)
print("Coefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"{feat}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:,.2f}")
print(f"R² Score: {r2:.4f}")

# Visualization: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.tight_layout()
plt.savefig("regression_result.png")
print("Chart saved as regression_result.png")