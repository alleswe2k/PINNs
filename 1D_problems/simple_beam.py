import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import OrderedDict

np.random.seed(1234)



class Setup():
    def __init__(self):
        pass

    def generate_points(self, n_points):
        x_t = np.random.uniform(0, 1, n_points - 2)
        x_t = np.append(x_t, [0, 1])
        x_t = np.sort(x_t).reshape(-1, 1)
        return x_t
    
    def set_bc(self, bc):
         


        

