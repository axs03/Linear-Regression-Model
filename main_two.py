import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load data from the main sheet to analyze its structure
data = pd.read_excel('stacks_time_data.xlsx', sheet_name='Sheet1')

# Preprocess Transaction Time to convert it into seconds since midnight
data['Transaction Time'] = pd.to_datetime(data['Transaction Time'])
data['Time in Seconds'] = (
    data['Transaction Time'].dt.hour * 3600 +
    data['Transaction Time'].dt.minute * 60 +
    data['Transaction Time'].dt.second
)

# Filter data to keep necessary columns and drop NaN values
filtered_data = data[['Revenue Center', 'Time in Seconds', 'Check Total']].dropna()

# Unique revenue centers
revenue_centers = filtered_data['Revenue Center'].unique()

# Dictionary to store results for each revenue center
results = {}

# Polynomial degree
degree = 3

# Process each revenue center
for center in revenue_centers:
    # Filter data for the current revenue center
    center_data = filtered_data[filtered_data['Revenue Center'] == center]
    X = center_data[['Time in Seconds']].values
    y = center_data['Check Total'].values

    # Polynomial transformation
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Fit polynomial regression
    model = LinearRegression()
    model.fit(X_poly, y)

    # Generate smooth predictions for plotting
    X_continuous = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    X_continuous_poly = poly.transform(X_continuous)
    y_continuous_pred = model.predict(X_continuous_poly)

    # Metrics
    y_pred = model.predict(X_poly)
    RSS = np.sum((y - y_pred) ** 2)
    TSS = np.sum((y - np.mean(y)) ** 2)
    SSE = TSS - RSS
    R2 = r2_score(y, y_pred)
    coefficients = model.coef_
    intercept = model.intercept_

    # Store results
    results[center] = {
        'RSS': RSS,
        'TSS': TSS,
        'SSE': SSE,
        'R2': R2,
        'Coefficients': coefficients.tolist(),
        'Intercept': intercept,
    }

    # Plot regression curve
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.5)
    plt.plot(X_continuous, y_continuous_pred, color='red', label=f'Polynomial Regression (Degree {degree})')
    plt.title(f'Polynomial Regression for {center}')
    plt.xlabel('Time in Seconds (since midnight)')
    plt.ylabel('Check Total')
    plt.legend()
    plt.grid(True)
    plt.show()

# Display results for each revenue center in a formatted way
for center, metrics in results.items():
    print(f"Results for {center}:")
    print(f"  RSS: {metrics['RSS']:.2f}")
    print(f"  TSS: {metrics['TSS']:.2f}")
    print(f"  SSE: {metrics['SSE']:.2f}")
    print(f"  R²: {metrics['R2']:.4f}")
    print(f"  Coefficients: {metrics['Coefficients']}")
    print(f"  Intercept: {metrics['Intercept']:.2f}")
    print("\n")
