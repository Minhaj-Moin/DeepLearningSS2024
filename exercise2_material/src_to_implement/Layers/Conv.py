import numpy as np
from scipy import signal
from scipy.ndimage import convolve1d
from  Layers.Base  import BaseLayer

np.prod((3,5,8)[1:]) * 3
class Conv(BaseLayer):
	def __init__(self, stride_shape, convolution_shape, num_kernels):
		BaseLayer.__init__(self)
		self.trainable=True
		self.weights = None
		self.bias = None
		self.stride_shape = stride_shape
		self.convolution_shape = convolution_shape
		self.num_kernels = num_kernels
	def initialize(self, weights_initializer, bias_initializer):
		# print("initialize CAlled")
		self.weights = weights_initializer.initialize((np.prod(self.convolution_shape[1:]), np.prod(self.convolution_shape)), np.prod(self.convolution_shape), np.prod(self.convolution_shape[1:]) * self.num_kernels)
		self.bias = bias_initializer.initialize((np.prod(self.convolution_shape[1:]), np.prod(self.convolution_shape)), np.prod(self.convolution_shape), np.prod(self.convolution_shape[1:])* self.num_kernels)

	def forward(self, input_tensor):
		'''shape (b, c, y) or (b, c, y, x)'''
		outshape = [input_tensor.shape[0], self.num_kernels, input_tensor.shape[2]]
		if len(input_tensor.shape)>2: outshape += [input_tensor.shape[3]]
		output = np.zeros(outshape)
		for ch in range(input_tensor.shape[1]):
			for ker in range(self.num_kernels):
				if self.convolution_shape == 2:
					output[:,ker] = convolve1d(input_tensor[:,ch], weights=self.weights[ker,ch]) + self.bias[ker]
				elif self.convolution_shape == 3:
					output[:,ker] = signal.correlate2d(input_tensor[ch,0], self.weights[ker,ch], boundary='fill', mode='same') + self.bias[ker]
		return output
