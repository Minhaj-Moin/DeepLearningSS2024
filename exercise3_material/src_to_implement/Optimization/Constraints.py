import numpy as np

class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
    def calculate_gradient(self, weights):
        return self.alpha * weights/np.abs(weights)
        pass
    def norm(self, weights):
        return self.alpha * np.abs(weights).sum()
        pass

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
    def calculate_gradient(self, weights):
        return self.alpha * weights
        pass
    def norm(self, weights):
        return self.alpha * np.power(weights,2).sum()
        pass
