import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

if "__main__" == __name__:
    # loading data
    data = pd.read_excel("stacks_data_2.xlsx")

    # Use Weekly Gross Sale as the feature and Monthly Gross Sale as the target
    features = ['Weekly Gross Sales']
    x = data[features]
    y = pd.to_numeric(data['Monthly Gross Sales'], errors='coerce')
    
    # Normalize the features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.5, random_state=42)
    
    # train the model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # predictions
    y_pred_test = model.predict(x_test)

    # Get beta0 (intercept) and beta1 (coefficient)
    beta0 = model.intercept_
    beta1 = model.coef_[0]

    # Calculate SSE (Sum of Squared Errors) for test set
    sse = np.sum((y_test - y_pred_test) ** 2)

    # Calculate SST (Total Sum of Squares)
    sst = np.sum((y_test - np.mean(y_test)) ** 2)

    # Calculate R-squared
    r_squared = r2_score(y_test, y_pred_test)

    print(f'beta0: {beta0}')
    print(f'beta1: {beta1}')
    print(f'SSE: {sse}')
    print(f'SST: {sst}')
    print(f'R-squared: {r_squared}')

    # Plot the actual data points
    plt.scatter(scaler.inverse_transform(x_test), y_test, color='blue', label='Actual data')

    # Plot the regression line
    plt.plot(scaler.inverse_transform(x_test), y_pred_test, color='red', linewidth=2, label='Regression line')

    # Add labels and title
    plt.xlabel('Weekly Gross Sales')
    plt.ylabel('Monthly Gross Sales')
    plt.title('Regression Line vs Actual Data')
    plt.legend()

    # Show the plot
    plt.show()