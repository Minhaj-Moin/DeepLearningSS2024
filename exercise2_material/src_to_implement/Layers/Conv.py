import numpy as np
from scipy.signal import correlate, convolve


from Layers.Base import BaseLayer



class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels: int) -> None:
        BaseLayer.__init__(self)
        self.trainable = True

        #TODO initilize the paramters uniformly
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

           
        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)

        self.gradient_weights = None
        self.gradient_biases = None


    def forward(self, input_tensor: np.ndarray):
        self.input_tensor = input_tensor
        if len(self.convolution_shape) == 2:
            # 1D Convolution
            output_shape = (
                input_tensor.shape[0],
                self.num_kernels,
                input_tensor.shape[2]
            )
            output_tensor = np.zeros(output_shape)
            for i in range(self.num_kernels):
                for j in range(input_tensor.shape[1]):
                    output_tensor[:, i, :] += correlate(
                        input_tensor[:, j, :],
                        self.weights[i, j, :],
                        mode='same'
                    )
                output_tensor[:, i, :] += self.bias[i]
        elif len(self.convolution_shape) == 3:
            # 2D Convolution
            output_shape = (
                input_tensor.shape[0],
                self.num_kernels,
                input_tensor.shape[2],
                input_tensor.shape[3]
            )
            output_tensor = np.zeros(output_shape)
            for i in range(self.num_kernels):
                for j in range(input_tensor.shape[1]):
                    output_tensor[:, i, :, :] += correlate(
                        input_tensor[:, j, :, :],
                        self.weights[i, j, :, :],
                        mode='same'
                    )
                output_tensor[:, i, :, :] += self.bias[i]
        return output_tensor
    

    def backward(error_tensor):
        return

    def initialize(self, weights_initializer, bias_initializer):
        