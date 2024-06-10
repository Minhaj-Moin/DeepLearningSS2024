import numpy as np
class Conv(BaseLayer):
	def __init__(self, stride_shape, convolution_shape, num_kernels):
		BaseLayer.__init__(self)
		self.trainable=True
		self.weights = None
		self.bias = None
		self.stride_shape = stride_shape
		self.convolution_shape = convolution_shape
		self.num_kernels = num_kernels

	def forward(self, input_tensor):
		'''shape (b, c, y) or (b, c, y, x)'''
		