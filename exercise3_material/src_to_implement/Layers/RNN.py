if __name__ == "__main__":
    from Base import BaseLayer
else:
    from Layers.Base import BaseLayer
import numpy as np


class RNN(BaseLayer):

    def __init__(self,input_size, hidden_size, output_size):
        BaseLayer.__init__(self)
        self.hidden_state = np.zeros(hidden_size)
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.value = None
        self._memorize = False
        self._weights = None
        self.bias = None


    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size+1,self.output_size), self.input_size, self.output_size)
        self.bias = bias_initializer.initialize((self.output_size), 1, self.output_size)
    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    @property
    def memorize(self):
        return self._memorize
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
    @memorize.setter
    def memorize(self, state: bool):
        self._memorize = state

    def forward(self, input_tensor):
        exps = np.exp(input_tensor.T - np.max(input_tensor, axis=1)).T
        self.value = (exps.T / np.sum(exps, axis=1)).T
        return self.value

    def backward(self, error_tensor):
        return self.value * (
            (error_tensor.T - np.multiply(self.value, error_tensor).sum(axis=1).T).T
        )
