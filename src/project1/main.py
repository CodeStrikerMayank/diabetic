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
        """So we acutally need clip function to create deafult limits like minimum will be -500 and 500 
        to prevent unwated infinity or Nan errors """
        z = np.clip(z,-500,500) 
        return 1.0/(1.0+np.exp(-z))

    def neuron_brain_function(self):
        n  = len(self.x)
        for echo in range(self.echoes):
            z = np.dot(self.x,self.weights)+self.bias
            self.pred = self.sigmoid(z)
            error = self.pred - self.y
            dw = np.dot(self.x.T , error)/n 
            db = np.sum(error)/n

            self.weights -= dw * self.lrt
            self.bias -= db * self.lrt

            

a = artifical_neuron()
a.neuron_brain_function()
