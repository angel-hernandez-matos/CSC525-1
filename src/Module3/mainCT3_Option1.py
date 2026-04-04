# File: mainCT3.py
# Written by: Angel Hernandez
# Description: Module 3 - Critical Thinking - Option 1
# Requirement(s):
# For your assignment, you will build a linear regression model in Python.
# The Boston housing dataset can be loaded in scikit-learn using the command load_boston() after from
# sklearn.datasets import load_boston.
# Using this data, our model should be able to predict the value of a house using the features given in the dataset.

import os
import sys
import subprocess
import requests
from io import StringIO

class DependencyChecker:
    @staticmethod
    def ensure_package(package_name):
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing missing package: {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' was installed successfully.")

class SimpleLinearRegression:
    def __init__(self):
        import numpy as np
        import pandas as pd
        from tqdm import tqdm as tqdm
        import matplotlib.pyplot as plt
        #from sklearn.datasets import load_boston as lb # ---> This was removed from scikit-learn since version 1.2
        from sklearn.linear_model import LinearRegression as lr
        from sklearn.model_selection import train_test_split as tts
        from sklearn.metrics import mean_squared_error as mse, r2_score as r2_score
        self.__pd = pd
        self.__np = np
        self.__mse = mse
        self.__plt = plt
        self.__tqdm = tqdm
        self.__y_prediction = None
        self.__r2_score = r2_score
        self.__train_test_split = tts
        self.__linear_regression = lr
        self.__load_dataset()

    def __load_dataset(self):
        buffer = ""
        url = "http://lib.stat.cmu.edu/datasets/boston"
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        for chunk in self.__tqdm(response.iter_content(chunk_size=1024), total=total // 1024, desc="Downloading boston dataset..."):
            buffer += chunk.decode('utf-8')
        raw_df = self.__pd.read_csv(StringIO(buffer), sep=r"\s+", skiprows=22, header=None)
        self.__data = self.__np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        self.__target = raw_df.values[1::2, 2]
        self.X = self.__data # Feature matrix
        self.Y = self.__target # Target vector

    def __train(self):
        self.X_simple = self.X[:, self.__np.newaxis, 5] # Using RM (average number of rooms per dwelling) which is index 5
        x_train, x_test, y_train, y_test = self.__train_test_split(self.X_simple, self.Y, test_size=0.2, random_state=42)
        self.__test = x_test
        self.__y_pred = y_test
        self.model = self.__linear_regression()
        self.model.fit(x_train, y_train)

    def __show_plot(self):
        fig = self.__plt.figure()
        fig.canvas.manager.set_window_title("Module 3 - Critical Thinking - Option 1 -  Simple Linear Regression in Scikit Learn")
        self.__plt.scatter(self.__test, self.__y_pred, color="green", label="Actual Prices")
        self.__plt.plot(self.__test, self.__y_prediction, color="red", linewidth=2, label="Predicted Regression Line")
        self.__plt.xlabel("Average Number of Rooms (RM)")
        self.__plt.ylabel("Housing Price ($1000s)")
        self.__plt.title("Simple Linear Regression: RM vs. Price")
        self.__plt.legend()
        self.__plt.show()

    def predict(self):
        self.__train()
        self.__y_prediction = self.model.predict(self.__test)
        # Evaluate the model
        mse = self.__mse(self.__y_pred, self.__y_prediction)
        r2 = self.__r2_score(self.__y_pred, self.__y_prediction)
        print("Simple Linear Regression on Boston Housing Dataset")
        print("**************************************************")
        print(f"Coefficient (slope): {self.model.coef_[0]:.4f}")
        print(f"Intercept: {self.model.intercept_:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}\n")
        self.__show_plot()

class TestCaseRunner:
    @staticmethod
    def run_test():
        linear_regression = SimpleLinearRegression()
        linear_regression.predict()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        dependencies = ['numpy', 'pandas', 'scikit-learn','matplotlib', 'tqdm']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 3 - Critical Thinking | Option 1 ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__': main()