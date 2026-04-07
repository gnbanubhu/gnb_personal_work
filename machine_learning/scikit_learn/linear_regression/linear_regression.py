from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# ── Sample Data ───────────────────────────────────────────────────────────────
# House size (sq ft) vs Price ($)
X = np.array([[750], [1000], [1200], [1500], [1800], [2000], [2500], [3000]])
y = np.array([150000, 200000, 230000, 280000, 330000, 370000, 450000, 540000])

# ── Split Data ────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train Model ───────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

print("=" * 50)
print("LINEAR REGRESSION - HOUSE PRICE PREDICTION")
print("=" * 50)
print(f"Coefficient  : {model.coef_[0]:.2f}  (price per sq ft)")
print(f"Intercept    : {model.intercept_:.2f}")
print(f"R² Score     : {r2_score(y_test, y_pred):.4f}")
print(f"MSE          : {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE         : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# ── Predict ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("PREDICTIONS")
print("=" * 50)
for size in [1100, 1600, 2200]:
    price = model.predict([[size]])[0]
    print(f"House size {size} sq ft => Predicted price: ${price:,.0f}")
