import numpy as np
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

		num_strides = self.stride_shape

		if type(num_strides) == int:
			num_strides = tuple((num_strides * np.ones(len(a.shape))).astype(int))


		sub_shape = self.convolution_shape
		a = input_tensor
		# view_shape = tuple(np.subtract(a.shape, sub_shape) + 2 - (num_strides)) + sub_shape
		# print("view_shape",view_shape)
		# strides = a.strides + a.strides
		# strides = np.array(strides)
		# for k in range(len(a.strides)):
		# 	strides[k] *= num_strides[k]

		# sub_matrices = np.lib.stride_tricks.as_strided(a,view_shape,strides)
		conv_filter = self.weights
		# m = np.einsum('ij,ijkl->kl',conv_filter,sub_matrices)
		# # print(sub_matrices)
		# print(m)
		from scipy import signal
		shp = [input_tensor.shape[0],self.num_kernels, input_tensor.shape[2]]
		if len(input_tensor.shape)>2: shp += [input_tensor.shape[3]]
		c = np.zeros(shp)
		# for i in range(input_tensor.shape[0]):
		# 	print(i, input_tensor[i].shape)
		# 	for ch in range(self.num_kernels):
		# 		if self.convolution_shape == 2:
		# 			from scipy.ndimage import convolve1d
		# 			c[i,ch,:] = convolve1d(input_tensor[i,0], weights=conv_filter[])
		# 		else:
		# 			c[i,ch,:] = signal.correlate2d(input_tensor[i,0], conv_filter, boundary='fill', mode='same')
		for i in range(input_tensor.shape[1]):
			# print(i, input_tensor[i].shape)
			for ch in range(self.num_kernels):
				if self.convolution_shape == 2:
					from scipy.ndimage import convolve1d
					c[:,ch,:] = convolve1d(input_tensor[:,i], weights=conv_filter[ch,i]) + self.bias[ch]
				else:
					c[:,ch,:] = signal.correlate2d(input_tensor[i,0], conv_filter[ch,i], boundary='fill', mode='same') + self.bias[ch]
		return c
	# def forward(self, input_tensor: np.ndarray):
	# 	from scipy import signal
	# 	from scipy.ndimage import convolve1d
	# 	self.input_tensor = input_tensor
	# 	if len(self.convolution_shape) == 2:
	# 		# 1D Convolution
	# 		output_shape = (
	# 			input_tensor.shape[0],
	# 			self.num_kernels,
	# 			input_tensor.shape[2]
	# 		)
	# 		output_tensor = np.zeros(output_shape)
	# 		for i in range(self.num_kernels):
	# 			for j in range(input_tensor.shape[1]):
	# 				output_tensor[:, i, :] += convolve1d(
	# 					input_tensor[:, j, :],
	# 					weights=self.weights[i, j, :],
	# 					mode='same'
	# 				)
	# 			output_tensor[:, i, :] += self.bias[i]
	# 	elif len(self.convolution_shape) == 3:
	# 		# 2D Convolution
	# 		output_shape = (
	# 			input_tensor.shape[0],
	# 			self.num_kernels,
	# 			input_tensor.shape[2],
	# 			input_tensor.shape[3]
	# 		)
	# 		output_tensor = np.zeros(output_shape)
	# 		for i in range(self.num_kernels):
	# 			for j in range(input_tensor.shape[1]):
	# 				output_tensor[:, i, :, :] += signal.correlate2d(
	# 					input_tensor[:, j, :, :],
	# 					self.weights[i, j, :, :],
	# 					mode='same'
	# 				)
	# 			output_tensor[:, i, :, :] += self.bias[i]
	# 	return output_tensor