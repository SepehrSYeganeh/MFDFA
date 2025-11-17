import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(path: str) -> np.ndarray:
    raw_data = pd.read_csv(path)
    price = raw_data["price"].to_numpy()
    plt.plot(price)
    plt.title("BTC price")
    plt.show()
    return price