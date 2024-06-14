import numpy as np

# from Layers.Base import BaseLayer
from scipy import signal
import math
from copy import deepcopy


class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = np.array(stride_shape)
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True
        self.gradient_weights = None
        self.gradient_bias = None
        self.kernel = None

        self.weights = np.random.rand(self.num_kernels, *self.convolution_shape)
        self.bias = np.random.rand(self.num_kernels)
        self._optimizer = None  # two optimizers for weights and biases seperately

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        opt1 = deepcopy(opt)
        opt2 = deepcopy(opt)
        self._optimizer = [opt1, opt2]

    def forward(self, input_tensor):
        """
        The input layout for 1D is defined in b, c, y order,
        for 2D in b, c, y, x order.
        Here, b stands for the batch, c represents the channels and
        x, y represent the spatial dimensions.

        """
        self.last_input = input_tensor
        input_depth = input_tensor.shape[1]
        batch_size = input_tensor.shape[0]

        output = np.zeros((batch_size, self.num_kernels, *input_tensor.shape[2:]))
        for m in range(batch_size):
            for c in range(self.num_kernels):
                for d in range(input_depth):
                    output[m, c] += signal.correlate(
                        input_tensor[m, d], self.weights[c, d], "same"
                    )

                output[m, c] += self.bias[c]

        output = self.apply_stride(output)
        return output

    def apply_stride(self, output):
        """
        filters the output with respect to stride
        """
        if len(self.convolution_shape) == 3:
            return output[:, :, 0 :: self.stride_shape[0], 0 :: self.stride_shape[1]]
        else:
            return output[:, :, 0 :: self.stride_shape[0]]

    def upsample_stride(self, input_tensor, out_shape):

        result = np.zeros(out_shape)
        if len(out_shape) == 2:  # 2D
            result[0 :: self.stride_shape[0], 0 :: self.stride_shape[1]] = input_tensor
            return result
        else:
            result[0 :: self.stride_shape[0]] = input_tensor
            return result

    def pad(self, input_tensor):
        """'
        pads the input in such a way that correlation produces
        same shape output as the input tensor
        """
        # formula

        filters = np.array(self.convolution_shape[1:])
        padwh_before = np.floor(filters / 2).astype(int)
        # handle uneven
        padwh_after = filters - padwh_before - 1
        if len(filters) == 2:  # 2D
            pad_width = [
                (0, 0),
                (0, 0),
                (padwh_before[0], padwh_after[0]),
                (padwh_before[1], padwh_after[1]),
            ]
        else:
            pad_width = [(0, 0), (0, 0), (padwh_before[0], padwh_after[0])]

        padded_input = np.pad(input_tensor, pad_width=pad_width, constant_values=0)

        return padded_input

    def backward(self, error_tensor):
        """
        error tensor shape: dE/dout
        same shape as output -- which output, this layer or preceding layer

        """

        ## questions, where to apply stride
        ## how much to pad the input
        ## pad last input with zeros (such that conv of kernel gives same shape output)
        self.gradient_weights = np.zeros(self.weights.shape)
        if len(self.convolution_shape) == 3:
            axis = (0, 2, 3)
        else:
            axis = (0, 2)
        self.gradient_bias = np.sum(error_tensor, axis=axis)
        new_error_tensor = np.zeros(self.last_input.shape)
        input_depth = self.last_input.shape[1]
        padded_input = self.pad(self.last_input)

        for m in range(self.last_input.shape[0]):
            for i in range(self.num_kernels):
                for j in range(input_depth):
                    # upsame the error tensor according to stride
                    upsampled_error = self.upsample_stride(
                        error_tensor[m, i], self.last_input.shape[2:]
                    )
                    out_w = signal.correlate(
                        padded_input[m, j], upsampled_error, "valid"
                    )
                    self.gradient_weights[i, j] += out_w

                    out_err = signal.convolve(
                        upsampled_error, self.weights[i, j], "same"
                    )
                    new_error_tensor[m, j] += out_err

        if self.optimizer != None:
            self.weights = self.optimizer[0].calculate_update(
                self.weights, self.gradient_weights
            )
            self.bias = self.optimizer[1].calculate_update(
                self.bias, self.gradient_bias
            )

        return new_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = weights_initializer.initialize(
            self.weights.shape, fan_in, fan_out
        )
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)
