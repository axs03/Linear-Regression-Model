import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import regression_model as rm


if "__main__" == __name__:
    # load data
    data = pd.read_excel("stacks_data.xlsx")

    # converting to datetime and then into minutes after midnight
    data['Quarter-Hour'] = pd.to_datetime(data['Quarter-Hour'], format='%H:%M %p')
    x = data['Quarter-Hour'].dt.hour * 60 + data['Quarter-Hour'].dt.minute
    
    print(data['Quarter-Hour'])

    y = pd.to_numeric(data['Net Sales'], errors='coerce')
    
    # perform linear regression
    beta0, beta1, sse, sst, r_squared, y_pred = rm.getVariables(x, y)
    
    print(f"Intercept (beta0): {beta0:.3f}")
    print(f"Slope (beta1): {beta1:.3f}")
    print(f"SSE: {sse:.3f}")
    print(f"SST: {sst:.3f}")
    print(f"R-squared: {r_squared:.3f}\n")

    # plot the data
    plt.scatter(x, y, label='Actual', color='blue')
    plt.plot(x, y_pred, label='Predicted', color='red')
    plt.xlabel('Day Minutes since Midnight')
    plt.ylabel('Net Sales')
    plt.legend()
    plt.title('Net Sales vs. Time for last 7 days')
    plt.show()
    