import pandas as pd
import numpy as np

def getVariables(x, y):
    n = len(x)
    
    # means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # covariance and variance
    covariance = np.sum((x - x_mean) * (y - y_mean))
    variance_x = np.sum((x - x_mean) ** 2)
    
    # coefficients
    beta1 = covariance / variance_x
    beta0 = y_mean - beta1 * x_mean
    
    # predictions
    y_pred = beta0 + beta1 * x
    
    # SSE (Sum of Squared Errors)
    sse = np.sum((y - y_pred) ** 2)
    
    # SST (Total Sum of Squares)
    sst = np.sum((y - y_mean) ** 2)
    
    # R-Squared
    r_squared = 1 - (sse / sst)
    
    return beta0, beta1, sse, sst, r_squared, y_pred