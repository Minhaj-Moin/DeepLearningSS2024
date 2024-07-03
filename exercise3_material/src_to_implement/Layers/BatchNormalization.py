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
        self.test_mean = 0
        self.test_var = 0
        self.X_hat = None
    def initialize(self):
        self.weights = np.ones((self.channels,))
        self.bias = np.zeros((self.channels,))

    def forward(self, input_tensor):
        alpha = 0.8
        self.input_tensor = input_tensor
        X_hat = np.zeros(input_tensor.shape)
        if not self.testing_phase:
            self.input_tensor_shape = input_tensor.shape
            # print(input_tensor.shape, input_tensor.mean(axis=1).shape)
            # X_hat = np.zeros(input_tensor.shape)
            # print(input_tensor.mean(axis=1, keepdims=True))
            for b in range(input_tensor.shape[0]):
                if len(input_tensor.shape) > 2:
                    for c in range(input_tensor.shape[1]):
                        self.test_mean = alpha * self.test_mean + (1-alpha) * input_tensor[b,c].flatten().mean(axis=0)
                        self.test_var = alpha * self.test_var + (1-alpha) * input_tensor[b,c].flatten().var(axis=0)
                        X_hat[b,c] = (input_tensor[b,c] - input_tensor[b,c].flatten().mean(dtype='float64'))/np.sqrt(input_tensor[b,c].flatten().var(dtype='float64') + 1e-12)
                        # print(input_tensor[b,c], input_tensor[b,c].flatten().mean(dtype='float64'))
                else:
                    self.test_mean = alpha * self.test_mean + (1-alpha) * input_tensor[b].mean()
                    self.test_var = alpha * self.test_var + (1-alpha) * input_tensor[b].var()
                    X_hat[b] = (input_tensor[b] - input_tensor[b].flatten().mean(dtype='float64'))/np.sqrt(input_tensor[b].flatten().var(dtype='float64') + 1e-12)
                    # print(b, X_hat[b].mean())

            if len(input_tensor.shape) == 2:
                X_hat = (input_tensor - input_tensor.mean(axis=0, keepdims=True))/np.sqrt(input_tensor.var(axis=0, keepdims=True) + 1e-12)
                self.X_hat = X_hat
                return self.weights * X_hat + self.bias
            self.X_hat = X_hat
            return self.reformat(self.weights * self.reformat(X_hat) + self.bias)
        else:
            # X_hat = (input_tensor - self.test_mean)/np.sqrt(self.test_var + 1e-12)
            # if len(input_tensor.shape) == 2: return self.weights * () + self.bias
            for b in range(input_tensor.shape[0]):
                if len(input_tensor.shape) == 2:
                    X_hat[b] = (input_tensor[b] - self.test_mean)/np.sqrt(self.test_var*1.2 + 1e-12)
                    # print(b ,input_tensor[b], input_tensor[b] - self.test_mean)
                else:
                    for c in range(input_tensor.shape[1]):
                        X_hat[b,c] = (input_tensor[b,c] - self.test_mean)/np.sqrt(self.test_var + 1e-12)
            # print(self.test_mean/np.sqrt(self.test_var + 1e-8))
            return self.weights * X_hat + self.bias if len(input_tensor.shape) == 2 else self.reformat(self.weights * self.reformat(X_hat) + self.bias)
            # return self.reformat(self.weights * self.reformat(X_hat) + self.bias)


    def reformat(self, tensor):
        if len(tensor.shape) == 2:
            b , h , m, n = self.input_tensor_shape
            return tensor.reshape(b, m, n, h).transpose(0,3,1,2)
        else:
            b , h , m, n = tensor.shape
            return tensor.reshape(b, h, m * n).transpose(0,2,1).reshape(b * m * n, h)

    def backward(self, error_tensor):
        E_t = error_tensor
        X_hat = self.X_hat
        input_tensor = self.input_tensor
        if len(error_tensor.shape) > 2:
            E_t = self.reformat(error_tensor)
            X_hat = self.reformat(self.X_hat)
            input_tensor = self.reformat(self.input_tensor)
        self.gradient_weights = (E_t * X_hat).sum(axis=0)
        self.gradient_bias = E_t.sum(axis=0)
        if len(error_tensor.shape) > 2:
            zz = compute_bn_gradients(E_t, input_tensor, self.weights, input_tensor.mean(axis=0), input_tensor.var(axis=0))
            print(zz.shape)
            return self.reformat(zz)
        return compute_bn_gradients(E_t, input_tensor, self.weights, input_tensor.mean(axis=0), input_tensor.var(axis=0))
