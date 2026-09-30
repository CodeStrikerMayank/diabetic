import pandas as pd
import numpy as np 

class file_management:

    def __init__(self):
        self.data = pd.read_csv("diabetes.csv")
        features = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
        self.x = self.data[features].to_numpy()
        self.y = self.data[["Outcome"]].to_numpy()
        
class artifical_neuron(file_management):

    def __init__(self):
        super().__init__()
        self.weights = np.zeros((8,1))
        self.bias = 0
        self.echoes = 100000
        self.lrt = .001

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def neuron_brain_function(self):
        for echo in range(self.echoes):
            z = np.dot(self.x,self.weights)+self.bias
            self.pred = self.sigmoid(z)
            error = self.pred - self.y
            dw = np.mean(self.x.T @ error)
            db = np.mean(self.x)
