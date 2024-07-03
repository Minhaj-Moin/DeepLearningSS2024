if __name__ == "__main__":
    from Base import BaseLayer
else:
    from Layers.Base import BaseLayer
    from Layers.Helpers import compute_bn_gradients
import numpy as np


class BatchNormalization(BaseLayer):

    def __init__(self, channels):
        BaseLayer.__init__(self)
        self.channels = channels
        self.trainable = True
        self.testing_phase = False
        self.initialize()
        self.test_mean = 0  #np.zeros(channels)
        self.test_var = 0  #np.zeros(channels)
        self.X_hat = None
        self._optimizer = None
        self._gradient_bias = None
        self._gradient_weights = None

    def initialize(self):
        self.weights = np.ones((self.channels,))
        self.bias = np.zeros((self.channels,))

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

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias

    def forward(self, input_tensor):
        alpha = 0.000001
        self.input_tensor = input_tensor.copy()
        if self.input_tensor.ndim > 2: input_tensor = self.reformat(input_tensor)
        print(self.input_tensor.shape)
        X_hat = np.zeros_like(input_tensor)

        if not self.testing_phase:
            self.test_mean = alpha * self.test_mean + (1 - alpha) * input_tensor.mean(axis=0)
            self.test_var = alpha * self.test_mean + (1 - alpha) * input_tensor.var(axis=0)
            X_hat = (input_tensor - input_tensor.mean(axis=0)) / np.sqrt(input_tensor.var(axis=0) + 1e-12)
            self.X_hat = X_hat
            out = self.weights * X_hat + self.bias
            if self.input_tensor.ndim > 2:
                self.X_hat = self.reformat(X_hat)
                out = self.reformat(out)
            return out
        else:
            for b in range(input_tensor.shape[0]):
                if len(input_tensor.shape) == 2:
                    X_hat[b] = (input_tensor[b] - self.test_mean)/np.sqrt(self.test_var + 1e-12)
                else:
                    for c in range(input_tensor.shape[1]):
                        X_hat[b, c] = (input_tensor[b, c] - self.test_mean[b,c]) / np.sqrt(self.test_var + 1e-12)
            return self.weights * X_hat + self.bias if len(input_tensor.shape) == 2 else self.reformat(self.weights * self.reformat(X_hat) + self.bias)

    def reformat(self, tensor):
        if len(tensor.shape) == 2:
            b, h, m, n = self.input_tensor.shape
            return tensor.reshape(b, m, n, h).transpose(0, 3, 1, 2)
        else:
            b, h, m, n = tensor.shape
            return tensor.reshape(b, h, m * n).transpose(0, 2, 1).reshape(b * m * n, h)

    def backward(self, error_tensor):
        E_t = error_tensor
        X_hat = self.X_hat
        input_tensor = self.input_tensor
        if len(error_tensor.shape) > 2:
            E_t = self.reformat(error_tensor)
            X_hat = self.reformat(self.X_hat)
            input_tensor = self.reformat(self.input_tensor)
            print(E_t.shape, X_hat.shape, input_tensor.shape, error_tensor.shape, self.X_hat.shape, self.input_tensor.shape)
        self.gradient_weights = (E_t * X_hat).sum(axis=0)
        self.gradient_bias = E_t.sum(axis=0)
        if len(error_tensor.shape) > 2:
            return self.reformat(compute_bn_gradients(E_t, input_tensor, self.weights, input_tensor.mean(axis=0),
                                                      input_tensor.var(axis=0)))
        out = compute_bn_gradients(E_t, input_tensor, self.weights, input_tensor.mean(axis=0), input_tensor.var(axis=0))
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(
                self.weights, self._gradient_weights
            )
        return out
