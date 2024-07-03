import numpy as np
from Layers.Base import BaseLayer
class TanH(BaseLayer):
    def __init__(self):
        BaseLayer.__init__(self)
        self.trainable = False
    def forward(self, input_tensor):
        self.value = np.tanh(input_tensor)
        return self.value

    def backward(self, error_tensor):
        return error_tensor * (1 - np.power(self.value,2))