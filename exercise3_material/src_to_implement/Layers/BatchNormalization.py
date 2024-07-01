if __name__ == "__main__":
    from Base import BaseLayer
else:
    from Layers.Base import BaseLayer
import numpy as np


class BatchNormalization(BaseLayer):

    def __init__(self):
        BaseLayer.__init__(self)
        self.value = None

    def forward(self, input_tensor):
        exps = np.exp(input_tensor.T - np.max(input_tensor, axis=1)).T
        self.value = (exps.T / np.sum(exps, axis=1)).T
        return self.value

    def backward(self, error_tensor):
        return self.value * (
            (error_tensor.T - np.multiply(self.value, error_tensor).sum(axis=1).T).T
        )
