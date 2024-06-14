import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor

    def copy(self, deep=True):
        if deep:
            return Sgd(self.learning_rate)
        return self


class SgdWithMomentum:

    def __init__(self, learning_rate, momentum_rate=0.9):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.V = None
        pass

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.V is None:
            self.V = np.zeros_like(weight_tensor)
        self.V = (self.momentum_rate * self.V) - (self.learning_rate * gradient_tensor)
        print(self.V, weight_tensor, gradient_tensor)
        return weight_tensor + self.V
    def copy(self, deep = True):
        return SgdWithMomentum(self.learning_rate, self.momentum_rate)

class Adam:
    def __init__(self, learning_rate=0.001, mu=0.9, rho=0.999):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-8

        self.V = None
        self.R = None
        self.k = 0
        pass

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.V is None:
            self.V = np.zeros_like(weight_tensor)

        if self.R is None:
            self.R = np.zeros_like(weight_tensor)

        self.V = self.mu * self.V + (1 - self.mu) * gradient_tensor
        self.R = self.rho * self.R + (1 - self.rho) * gradient_tensor**2

        self.k += 1.0

        v_hat = self.V / (1 - self.mu**self.k)
        r_hat = self.R / (1 - self.rho**self.k)

        return weight_tensor - self.learning_rate * (
            v_hat / (np.sqrt(r_hat) + self.epsilon)
        )
    def copy(self, deep = True):
        return Adam(self.learning_rate, self.mu, self.rho)