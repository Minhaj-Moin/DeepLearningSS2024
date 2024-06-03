import numpy as np

class Constant:
    def __init__(self, value=0.1):
        self.value = value
    def initialize(self, weights_shape, fan_in, fan_out):
        return self.value * np.ones(weights_shape)

class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.random(weights_shape)

class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(0, np.sqrt(2)/np.sqrt(fan_out+fan_in), weights_shape)

class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(0,np.sqrt(2/fan_in), weights_shape)
