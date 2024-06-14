import numpy as np
from scipy.signal import correlate, convolve


from Layers.Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        BaseLayer.__init__(self)
        self.trainable = True

        self.stride_shape = np.array(stride_shape)
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.weights = np.random.rand(self.num_kernels, *self.convolution_shape)
        self.bias = np.random.rand(self.num_kernels)

        self._optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(
            self.convolution_shape[1:]
        )  # NOTE why this shape??

        self.weights = weights_initializer.initialize(
            self.weights.shape, fan_in, fan_out
        )
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer


    def forward(self, input_tensor):
        """input_tensor shape (b, c, y) or (b, c, y, x)"""

        self.batch_size = input_tensor.shape[0]
        self.num_channels = input_tensor.shape[1]
        self.input_dim = input_tensor.shape[2:]
        self.input_tensor = input_tensor

        output = np.zeros((self.batch_size, self.num_kernels, *self.input_dim))

        for b in range(self.batch_size):
            for k in range(self.num_kernels):  # the number of kernals determine the number of channels for the output
                for c in range(self.num_channels):
                    output[b, k] += correlate(
                        input_tensor[b, c], self.weights[k, c], mode="same"
                    )  # accumulating the convolution process on the basis of channels
                output[b, k] += self.bias[k]  # for every kernal a scalar value of a bias is added

        self.input_tensor = input_tensor

        # Applying Stride
        if len(self.convolution_shape) == 2:
            return output[:, :, 0 :: self.stride_shape[0]]

        if len(self.convolution_shape) == 3:
            return output[:, :, 0 :: self.stride_shape[0], 0 :: self.stride_shape[1]]


    def backward(self, error_tensor):

        self.gradient_weights = np.zeros(self.weights.shape)
        prev_error_tensor = np.zeros(self.input_tensor.shape)

        input_tensor_padded =  self.__pad(self.input_tensor)

        for b in range(self.batch_size):
            for k in range(self.num_kernels):
                for c in range(self.num_channels):
                    upsampled_error_tensor = self.__upsample(error_tensor[b,k],self.input_tensor.shape[2:])
                    prev_error_tensor[b, c] += convolve(
                        upsampled_error_tensor, self.weights[k, c], mode="same"
                    )
                    self.gradient_weights[k, c] += correlate(
                        input_tensor_padded[b, c], upsampled_error_tensor, mode="valid"
                    )
                    
        self.gradient_bias = np.sum(error_tensor,
                                    axis=(0,2,3) if (len(self.convolution_shape)==3) else (0,2)
                                    )

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)

        return prev_error_tensor
    

    def __pad(self, input_tensor):
        
        pad_before = np.floor(np.array(self.convolution_shape[1:]) / 2).astype(int)
        pad_after = np.array(self.convolution_shape[1:]) - pad_before - 1
        
        if np.array(self.convolution_shape[1:]).size == 2: 
            pad_width = [
                (0, 0),  # No padding for batch dimension
                (0, 0),  # No padding for channel dimension
                (pad_before[0], pad_after[0]),  # Padding for height
                (pad_before[1], pad_after[1])   # Padding for width
            ]
        else:
            pad_width = [
                (0, 0),  # No padding for batch dimension
                (0, 0),  # No padding for channel dimension
                (pad_before[0], pad_after[0])   # Padding for 1D case
            ]

        return np.pad(input_tensor, pad_width=pad_width, constant_values=0)
    

    def __upsample(self, input_tensor, out_shape):
        
        upsampled_result = np.zeros(out_shape)
        
        if len(out_shape) == 2: 
            stride_h, stride_w = self.stride_shape
            upsampled_result[::stride_h, ::stride_w] = input_tensor
        else:  
            stride = self.stride_shape[0]
            upsampled_result[::stride] = input_tensor
        
        return upsampled_result
