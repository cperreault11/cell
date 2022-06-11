"""
Data structure is important. For each cell at timestep t, need access to all cells
 within epsilon at timesteps t-i, i = 1...t-1
for each of these neighboring cells, need access to the events at each timestep
"""
import numpy as np
import pandas as pd
import torch


class CPP:
    """
    Class to represent a learned cellular point process
    """
    def __init__(self):
        """
        Initialize all the parameters to be fit
        """
        self.eps = 0.06
        self.adivdiv = -1
        self.adeldel = -1
        self.adivdel = 1
        self.adeldiv = 1
        self.mudiv = 1
        self.mudel = 1
        self.bdivdiv = 1
        self.bdeldel = 1
        self.bdivdel = 1
        self.bdeldiv = 1

    def loglik(self):
        """
        Get the log likelihood of the dataset given parameters 
        """
        