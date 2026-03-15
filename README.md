# Linear Regression — Predicting Car Prices

## Project Summary

Built a linear regression model to predict car prices using a real-world dataset from Hugging Face. This was my first end-to-end ML project — covering data loading, exploratory data analysis, feature engineering, model training, and evaluation.

**Dataset:** [jayaprakash-m/linearRegressionDS](https://huggingface.co/datasets/jayaprakash-m/linearRegressionDS)

**Business Problem:** Can we predict the price of a used car based on its features — so buyers and sellers can make informed pricing decisions?

---

## What I Built

A linear regression model that predicts car `Price` from features like `Year`, `Kilometer`, and other numeric attributes.

---

## Key Learnings

### 1. What is Linear Regression?

Linear regression finds the best straight line through data to predict a continuous number.

```
Price = (Slope × Year) + Intercept
```

- **Slope** — how much price changes per unit increase in the feature
- **Intercept** — the base price when all features are zero
- **R² Score** — how much of the price variation is explained by the model (closer to 1.0 = better)

### 2. Real World Data vs Synthetic Data

Started with synthetic (generated) data:
```python
house_price = (house_size * 150) + 50000 + noise
```
Got R² = 0.97 because the relationship was perfect by design.

Moved to a real car dataset — R² dropped significantly because real data has messy, complex relationships that a single feature can't fully explain.

**Lesson:** High R² on synthetic data means nothing. Real world data is always messier.

### 3. Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- 80% of data used for training
- 20% used for testing — data the model has never seen
- R² is only meaningful when measured on the test set

**Analogy:** Like a teacher giving students 80 practice questions to study, then testing them on 20 new questions. Scoring 100% on questions you already studied proves nothing.

### 4. Single Feature vs Multiple Features

Started with just `Year` as a feature:
```
R² = 0.09 — Year alone explains only 9% of price variation
```

Year is just one small factor in car pricing. Engine size, kilometers driven, brand all matter more.

**Lesson:** Feature selection is critical. More relevant features = better model. This is where PM thinking helps — asking "what actually drives this outcome?"

### 5. Real World Data Cleaning Challenges

The dataset had text mixed into numeric columns:
- `Engine` column had values like `1500cc`
- `Max Power` had values like `100bhp`

Linear regression only understands numbers — these columns needed cleaning before use.

**Fix:** Used `select_dtypes` to automatically filter to numeric columns only:
```python
df_numeric = df.select_dtypes(include=[np.number])
```

### 6. Evaluation Metrics

**R² Score (R-squared)**
```
R² = 0.97 → model explains 97% of price variation (synthetic data)
R² = 0.09 → model explains only 9% of price variation (Year feature only)
```

**Mean Squared Error (MSE)**
- Average of squared differences between predicted and actual prices
- Lower is better
- Hard to interpret directly — use R² for intuition

### 7. Visualisation

Two key charts for linear regression:

**Regression fit plot** — shows the red prediction line through blue actual data points. A good fit means the line runs through the middle of the data cloud.

**Actual vs Predicted plot** — plots actual prices on X axis and predicted prices on Y axis. A perfect model = all points on the diagonal line.

### 8. Effect of Noise on Model Performance

```python
# Low noise → R² = 0.97 (model finds signal easily)
noise = np.random.randn(100, 1) * 20000

# High noise → R² drops significantly (signal buried in noise)
noise = np.random.randn(100, 1) * 200000
```

**Lesson:** R² drops as noise increases. Real world datasets have a lot of noise — this is normal and expected.

---

## Model Performance

| Features Used | R² Score |
|---|---|
| Year only | 0.09 |
| Year + Kilometer | TBD — next iteration |

---

## Tools & Libraries

```python
datasets          # Hugging Face dataset loading
pandas            # Data manipulation
numpy             # Numerical operations
scikit-learn      # Model training and evaluation
matplotlib        # Visualisation
python-dotenv     # Environment variable management
huggingface_hub   # Authentication
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/shambhavichaugule/linear-regression-project.git
cd linear-regression-project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Add your Hugging Face token to .env
echo "HF_TOKEN=your_token_here" > .env

# Run the model
python linear_regression.py
```

---

## PM Perspective

This project simulates a real product decision: **should we build a car price prediction feature?**

**Before building:**
- Understand what features actually drive the outcome (Year alone is weak)
- Clean messy data before modeling — text in numeric columns breaks everything
- Separate train and test data before evaluating — otherwise you're measuring memorisation not learning

**In production:**
- R² of 0.09 with just Year means this model is not production ready
- Need more features: engine size, kilometers, brand, transmission type
- A simple heuristic (average price by make and year) might outperform this model — always check before shipping

**Key insight:** A model is only as good as the features you give it. Feature engineering is a product decision as much as a technical one — you need domain knowledge to know what drives car prices.

---

## Next Steps

- Clean text columns (Engine, Max Power) to extract numeric values
- Add multiple features to improve R²
- Compare performance of single feature vs multiple features
- Try polynomial regression for non-linear relationships

---
