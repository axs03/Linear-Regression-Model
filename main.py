import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import regression_model as rm


if "__main__" == __name__:
    # load data
    data = pd.read_csv("stacks_data.csv")

    x = np.arange(len(data))
    y = data['TOTALSA'].to_numpy()
    
    # perform linear regression
    beta0, beta1, sse, sst, r_squared, y_pred = rm.getVariables(x, y)
    
    print(f"Intercept (beta0): {beta0:.3f}")
    print(f"Slope (beta1): {beta1:.3f}")
    print(f"SSE: {sse:.3f}")
    print(f"SST: {sst:.3f}")
    print(f"R-squared: {r_squared:.3f}\n")

    # plot the data
    plt.plot(x, y, label='Actual')
    plt.plot(x, y_pred, label='Predicted')
    plt.legend()
    plt.show()
    