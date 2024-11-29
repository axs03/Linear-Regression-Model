import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if "__main__" == __name__:
    # load
    data = pd.read_csv("data.csv")

    # plot
    plt.plot(data["x"], data["y"])
    plt.show()