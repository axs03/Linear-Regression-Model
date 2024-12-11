import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import regression_model as rm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

if "__main__" == __name__:
    # loading data
    data = pd.read_excel("stacks_data.xlsx")

    # converting to datetime and then into minutes after midnight
    data['Quarter-Hour'] = pd.to_datetime(data['Quarter-Hour'], format='%H:%M %p')
    data['Minutes'] = data['Quarter-Hour'].dt.hour * 60 + data['Quarter-Hour'].dt.minute
    
    features = ['Minutes']  # can add more variables here
    x = data[features]
    y = pd.to_numeric(data['Net Sales'], errors='coerce')
    
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"After training")
    print(f"Train MSE: {train_mse:.3f}")
    print(f"Test MSE: {test_mse:.3f}")
    print(f"Train R-squared: {train_r2:.3f}")
    print(f"Test R-squared: {test_r2:.3f}\n")

    # plot the data
    plt.scatter(x['Minutes'], y, label='Actual', color='blue')
    plt.plot(x['Minutes'], model.predict(x), label='Predicted', color='red')
    plt.xlabel('Day Minutes since Midnight')
    plt.ylabel('Net Sales')
    plt.legend()
    plt.title('Net Sales vs. Time for last 7 days')
    plt.show()