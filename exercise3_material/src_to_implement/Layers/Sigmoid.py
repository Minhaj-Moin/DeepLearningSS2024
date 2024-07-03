if __name__ == "__main__":
    from Base import BaseLayer
else:
    from Layers.Base import BaseLayer
import numpy as np


class Sigmoid(BaseLayer):

    def __init__(self):
        BaseLayer.__init__(self)
        self.value = None
        self.trainable = False

    def forward(self, input_tensor):
        self.value = 1/(1 + np.exp(-input_tensor))
        return self.value

    def backward(self, error_tensor):
        return error_tensor * (self.value - np.power(self.value,2))
