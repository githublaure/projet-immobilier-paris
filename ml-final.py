import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
np.random.seed(42)
n_samples = 1000

# Generate features
surface = np.random.normal(60, 20, n_samples)  # Surface area in m²
rooms = np.random.randint(1, 6, n_samples)     # Number of rooms
distance_metro = np.random.normal(500, 200, n_samples)  # Distance to metro in meters
year = np.random.randint(1950, 2024, n_samples)  # Year of construction

# Generate target with some noise
base_price = 5000  # Base price per m²
price = (
    base_price * surface + 
    rooms * 10000 + 
    -0.1 * distance_metro + 
    (year - 1950) * 100 + 
    np.random.normal(0, 50000, n_samples)
)

# Create DataFrame
data = pd.DataFrame({
    'surface': surface,
    'rooms': rooms,
    'distance_metro': distance_metro,
    'year': year,
    'price': price
})

# Split features and target
X = data.drop('price', axis=1)
y = data['price']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RidgeCV(alphas=np.logspace(-10, 10, 21))
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"RMSE: {rmse:,.2f} €")
print(f"R²: {r2:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_
}).sort_values('coefficient', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Sample prediction
sample_house = pd.DataFrame({
    'surface': [75],
    'rooms': [3],
    'distance_metro': [400],
    'year': [2000]
})
sample_scaled = scaler.transform(sample_house)
predicted_price = model.predict(sample_scaled)[0]

print(f"\nSample Prediction:")
print(f"Predicted price for a {sample_house['surface'].iloc[0]}m² house:")
print(f"€{predicted_price:,.2f}")