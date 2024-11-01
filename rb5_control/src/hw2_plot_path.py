import pandas as pd
import matplotlib.pyplot as plt
import time

def plot_path():
    while True:
        data = pd.read_csv('/path/to/position_log.csv', names=['x', 'y', 'theta'])
        plt.clf()
        plt.plot(data['x'], data['y'], marker='o')
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Robot Path Tracking")
        plt.pause(1)

if __name__ == "__main__":
    plot_path()
