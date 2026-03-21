# File: mainCT2.py
# Written by: Angel Hernandez
# Description: Module 2 - Critical Thinking - Option 1
# Requirement(s):  KNN Classifier with Iris data. Your classifier should be able to, using this data,
# predict the type of iris based on the sepal length and width (the parts of the calyx) and the petal length and width,
# in centimeters. The Python program must accept input of 4 floating point numbers representing, respectively,
# sepal length, sepal width, petal length, and petal width.

import os
import csv
import math
from collections import Counter
from typing import Generic, TypeVar, Optional

T = TypeVar("T")

class ArgumentDefinition(Generic[T]):
    def __init__(self, name: str, default_value: T):
        self.name = name
        self.value: Optional[T] = None
        self.default_value = default_value
        self.caster = type(default_value)

    def read(self):
        value = input(f"{self.name} (default {self.default_value}): ")
        if value.strip() == "":
            self.value = self.default_value
        else:
            try:
                self.value = self.caster(value)
            except Exception:
                raise ValueError(f"Invalid value for {self.name}: {value}")

class KNNConfig:
    def __init__(self):
       self.sepal_width = 0.0
       self.petal_width = 0.0
       self.petal_length = 0.0
       self.sepal_length = 0.0
       self.csv_file = "iris.csv"
       arguments = [("sepal_length", ArgumentDefinition("Sepal Length", 5.1)),
                   ("sepal_width", ArgumentDefinition("Sepal Width", 3.5)),
                   ("petal_length", ArgumentDefinition("Petal Length", 1.4)),
                   ("petal_width", ArgumentDefinition("Petal Width", 0.2)),
                   ("csv_file", ArgumentDefinition("CSV File", self.csv_file))]

       for attr_name, arg in arguments:
           arg.read()
           setattr(self, attr_name, arg.value)

class KNNClassifier:
    def __init__(self, config = None):
      if config is None:
          config = KNNConfig()
      self.__config = config
      self.__data = self.__load_data()

    @staticmethod
    def __euclidean_distance(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def knn_predict(self, k=3):
        distances = []
        query = [self.__config.sepal_length, self.__config.sepal_width,
                 self.__config.petal_length, self.__config.petal_width]
        # Compute distance to every point
        for features, label in self.__data:
            dist = self.__euclidean_distance(features, query)
            distances.append((dist, label))
        # Sort by distance
        distances.sort(key=lambda x: x[0])
        # Select top k
        k_nearest = distances[:k]
        # Majority vote
        labels = [label for _, label in k_nearest]
        retval = Counter(labels).most_common(1)[0][0]
        return retval

    def __load_data(self):
        retval = []
        with open(self.__config.csv_file, "r") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                features = list(map(float, row[:4]))
                label = row[4]
                retval.append((features, label))
        return retval

class TestCaseRunner:
    @staticmethod
    def run_test():
        knn_classifier = KNNClassifier()
        prediction = knn_classifier.knn_predict() # K defaults to 3
        print(f"The predicted Iris species is '{ prediction }'")

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        clear_screen()
        print('*** Module 2 - Critical Thinking | Option 1 ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__': main()