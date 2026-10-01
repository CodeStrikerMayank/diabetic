import pandas as pd
import numpy as np 
import time
from rich.console import Console 
from rich.live import Live
console = Console()
from rich.panel import Panel
from rich.align import Align
import threading as td




"""File access and data extraction logic implementation here
    And algorithm"""
class file_management:

    def __init__(self):
        self.data = pd.read_csv("diabetes.csv")
        features = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
        self.x = self.data[features].to_numpy()
        self.y = self.data[["Outcome"]].to_numpy()

    def save_info(self):
        payload = [{"weights":self.weights,"bias":self.bias}]
        data = pd.DataFrame(payload)
        data.to_json("core.json",orient="records",indent=4)

    def file_logic(self):
        try:
            data = pd.read_json("core.json")
            self.weights = np.array(data["weights"][0])
            self.bias = data["bias"][0]
            return 
        except FileNotFoundError as err:
            print("File not found")
            self.weights = np.zeros((8,1))
            self.bias = 0
"""Artifical active neuron brian logic """
class artifical_neuron(file_management):


    def __init__(self):
        super().__init__()
        self.file_logic()
        self.echoes = 100000
        self.lrt = .001

    def accuracy_df(self):
        prediction = (self.pred>=0.5).astype(int)
        accruacy = np.mean(prediction==self.y)*100
        print("\033c")
        print(f"Current accurancy is :- { accruacy:.2f}%")
        time.sleep(.9)
            
    def sigmoid(self,z):
        """So we acutally need clip function to create deafult limits like minimum will be -500 and 500 
        to prevent unwated infinity or Nan errors """
        z = np.clip(z,-500,500) 
        return 1.0/(1.0+np.exp(-z))

    def neuron_background_regression_function(self):
        print("\033c")
        n  = len(self.x)
        for echo in range(self.echoes):
            z = np.dot(self.x,self.weights)+self.bias
            self.pred = self.sigmoid(z)
            error = self.pred - self.y
            dw = np.dot(self.x.T , error)/n 
            db = np.sum(error)/n
            
            self.weights -= dw * self.lrt
            self.bias -= db * self.lrt 
        self.accuracy_df()                 
        self.save_info()
            
"""output show me and in best way"""
a = artifical_neuron()
a.neuron_background_regression_function()