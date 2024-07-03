if __name__ == "__main__":
    from Base import BaseLayer
else:
    from Layers.Base import BaseLayer
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
                print(X_hat.mean(axis=0).shape, input_tensor.shape)
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
        if len(error_tensor.shape) > 2:
            error_tensor = self.reformat(error_tensor)
            self.X_hat = self.reformat(self.X_hat)
            self.input_tensor = self.reformat(self.input_tensor)
        dW = (error_tensor * self.X_hat).sum(axis=0)
        dB = error_tensor.sum(axis=0)
        # self.weights -= dW
        # self.bias -= dB
        self.gradient_bias = dB
        self.gradient_weights = dW
        dX_hat = error_tensor * self.weights
        dVar = (error_tensor * (self.input_tensor - self.input_tensor.mean(axis=0)) * (-1/2) * np.power(self.input_tensor.var(axis=0) + 1e-12, -3/2)).sum(axis=0)
        # print(self.input_tensor.var(axis=1)self.weights.shape)
        dMean = (dW * -1/np.sqrt(self.input_tensor.var(axis=0) + 1e-12)).sum(axis=0)
        zz1 = (dX_hat * 1/np.sqrt(self.input_tensor.var(axis=0) + 1e-12))
        zz2 = (dVar * 2 * (self.input_tensor - self.input_tensor.mean(axis=0))/self.input_tensor.shape[0])
        zz3 = (dMean * 1/self.input_tensor.shape[0])
        return zz1 + zz2 + zz3

