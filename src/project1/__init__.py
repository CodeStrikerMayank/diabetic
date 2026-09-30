import pandas as pd
import numpy as np 

class file_management:

    def __init__(self):
        self.data = pd.read_csv("diabetes.csv")
        features = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
        self.x = self.data[features].to_numpy()
        self.y = self.data[["Outcome"]].to_numpy()
        print(self.y.shape)

    def