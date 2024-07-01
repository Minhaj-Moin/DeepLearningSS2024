if __name__ == "__main__":
    from Base import BaseLayer
else:
    from Layers.Base import BaseLayer
import numpy as np


class Dropout(BaseLayer):

    def __init__(self, probability):
        BaseLayer.__init__(self)
        self.probability = probability
        self.mask_tensor = None
        self.testing_phase = False

    def forward(self, input_tensor):
        self.mask_tensor = np.random.binomial(size=input_tensor.shape, n=1, p=self.probability)
        if self.testing_phase: return input_tensor
        return (input_tensor * self.mask_tensor) * 1/self.probability

    def backward(self, error_tensor):
        if self.testing_phase: return error_tensor
        return error_tensor * self.mask_tensor * 1/self.probability