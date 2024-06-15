import numpy as np

if __name__ == "__main__":
    from Base import BaseLayer
else:
    from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        BaseLayer.__init__(self)
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self.bias = np.random.uniform(0, 1, (output_size))
        self.input_size = input_size
        self.output_size = output_size
        self.input_tensor = None

        self._optimizer = None
        self._gradient_weights = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    def forward(self, input_tensor):
        b = np.ones((input_tensor.shape[0], input_tensor.shape[1] + 1))
        b[:, :-1] = input_tensor
        self.input_tensor = b
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(
                self.weights, self._gradient_weights
            )
        return np.dot(error_tensor, self.weights.transpose())[:, :-1]

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(
            (self.input_size+1,self.output_size), self.input_size+1, self.output_size
        )
        self.bias = bias_initializer.initialize((self.output_size), 1, self.output_size)
