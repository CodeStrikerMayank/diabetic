from src.project1.main import file_management , artifical_neuron
import numpy as np
import pandas as pd


class est(file_management):

    def __init__(self):
        super().__init__()
        print(self.y.shape)
        print(self.x.shape)

class test(artifical_neuron):

    def __init__(self):
        super().__init__()
        self.neuron_brain_function()
        

a = test()
