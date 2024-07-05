from Layers.Base import BaseLayer
import numpy as np


class ReLU(BaseLayer):

    def __init__(self):
        BaseLayer.__init__(self)
        self.value = None

    def forward(self, input_tensor):
        self.value = np.abs((input_tensor > 0) * input_tensor)
        return self.value

    def backward(self, error_tensor):
        return error_tensor * np.abs((self.value > 0) * 1.0)
