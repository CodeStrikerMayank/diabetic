from src.project1.main import file_management
import numpy as np
import pandas as pd


class test(file_management):

    def __init__(self):
        super().__init__()
        print(self.y.shape)
        print(self.x.shape)


a = test()
