import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load data
data = pd.read_excel('stacks_time_data.xlsx', sheet_name='Sheet1')

# Preprocess Transaction Time to extract day of the week
data['Transaction Time'] = pd.to_datetime(data['Transaction Time'])
data['Day of Week'] = data['Transaction Time'].dt.day_name()  # e.g., 'Monday', 'Tuesday'

# Map days of the week to numbers
day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
data['Day Numeric'] = data['Day of Week'].map(day_mapping)

# Aggregate transaction counts by day of the week
transactions_per_day = data.groupby('Day Numeric').size()

# Prepare data for regression
X = np.array(transactions_per_day.index).reshape(-1, 1)  # Day of week as numeric
y = transactions_per_day.values  # Transaction counts

# Polynomial transformation
degree = 3
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# Fit polynomial regression
model = LinearRegression()
model.fit(X_poly, y)

# Generate smooth predictions
X_continuous = np.linspace(0, 6, 100).reshape(-1, 1)  # Days from 0 (Monday) to 6 (Sunday)
X_continuous_poly = poly.transform(X_continuous)
y_continuous_pred = model.predict(X_continuous_poly)

# Predict transaction counts for each day
predicted_counts = model.predict(poly.transform(np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)))
most_transactions_day = max(zip(day_mapping.keys(), predicted_counts), key=lambda x: x[1])

# Plot regression curve
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.7)
plt.plot(X_continuous, y_continuous_pred, color='red', label=f'Polynomial Regression (Degree {degree})')
plt.title("Predicted Transactions by Day of the Week")
plt.xlabel("Day of the Week (Numeric)")
plt.ylabel("Number of Transactions")
plt.xticks(ticks=list(day_mapping.values()), labels=list(day_mapping.keys()))
plt.legend()
plt.grid(True)
plt.show()

# Output the results
print("Predicted Transactions for Each Day:")
for day, count in zip(day_mapping.keys(), predicted_counts):
    print(f"  {day}: {count:.2f}")
    
print(f"\nThe day with the most predicted transactions is {most_transactions_day[0]} with {most_transactions_day[1]:.2f} transactions.")
