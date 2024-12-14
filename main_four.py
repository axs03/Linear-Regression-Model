import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

# Load data
data = pd.read_excel('stacks_time_data.xlsx', sheet_name='Sheet1')

# Preprocess Transaction Time to extract day of the week and hour of the day
data['Transaction Time'] = pd.to_datetime(data['Transaction Time'])
data['Day of Week'] = data['Transaction Time'].dt.day_name()  # e.g., 'Monday', 'Tuesday'
data['Hour'] = data['Transaction Time'].dt.hour  # Hour of the day (0-23)
data = data[(data['Hour'] >= 7) & (data['Hour'] <= 23)]  # Filter to only include transactions between 7 AM and 11 PM

# Map days of the week to numbers
day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
data['Day Numeric'] = data['Day of Week'].map(day_mapping)

# Prepare data: Group by day and hour, and count transactions
hourly_transactions = data.groupby(['Day Numeric', 'Hour']).size().reset_index(name='Transaction Count')

# Polynomial degree
degree = 3

# Dictionary to store results for each day
results = {}

# Process each day (0 = Monday, ..., 6 = Sunday)
for day_num in range(7):
    # Filter data for the current day
    day_data = hourly_transactions[hourly_transactions['Day Numeric'] == day_num]
    X = day_data[['Hour']].values  # Hour of the day
    y = day_data['Transaction Count'].values  # Transaction counts

    # Polynomial transformation
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Fit polynomial regression
    model = LinearRegression()
    model.fit(X_poly, y)

    # Generate smooth predictions for all hours (0-23)
    X_continuous = np.linspace(0, 23, 100).reshape(-1, 1)
    X_continuous_poly = poly.transform(X_continuous)
    y_continuous_pred = model.predict(X_continuous_poly)

    # Predict transaction counts for each hour (0-23)
    predicted_counts = model.predict(poly.transform(np.arange(24).reshape(-1, 1)))
    most_transactions_hour = np.argmax(predicted_counts)  # Hour with the highest predicted transactions
    most_transactions_count = predicted_counts[most_transactions_hour]

    # Calculate R^2 value
    y_pred = model.predict(X_poly)
    R2 = r2_score(y, y_pred)

    # Store results
    results[day_num] = {
        'Day': list(day_mapping.keys())[day_num],
        'Most Transactions Hour': most_transactions_hour,
        'Most Transactions Count': most_transactions_count,
        'R2': R2,
        'Model': model,
        'Predicted Counts': predicted_counts
    }

    # Plot regression curve with time labels from 7 AM to 11 PM
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.7)
    plt.plot(X_continuous, y_continuous_pred, color='red', label=f'Polynomial Regression (Degree {degree})')
    plt.title(f"Predicted Hourly Transactions for {list(day_mapping.keys())[day_num]} (R² = {R2:.4f})")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Number of Transactions")

    # Custom x-axis ticks and labels for 7 AM to 11 PM
    hours = np.arange(7, 24)
    labels = [f"{hour if hour <= 12 else hour - 12} {'AM' if hour < 12 else 'PM'}" for hour in hours]
    plt.xticks(ticks=hours, labels=labels, rotation=45)

    # Set x-axis range to 7 AM - 11 PM
    plt.xlim(7, 23)

    plt.legend()
    plt.grid(True)
    plt.show()

# Display results for each day
print("Predicted Peak Hourly Transactions for Each Day:")
for day_num, metrics in results.items():
    print(f"{metrics['Day']}:")
    print(f"  Hour with Most Transactions: {metrics['Most Transactions Hour']}:00")
    print(f"  Predicted Transactions: {metrics['Most Transactions Count']:.2f}")
    print(f"  R²: {metrics['R2']:.4f}\n")
