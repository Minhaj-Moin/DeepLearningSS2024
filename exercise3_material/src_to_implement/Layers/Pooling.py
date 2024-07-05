import numpy as np
if __name__=='__main__':
	from Base import BaseLayer
else: from Layers.Base import BaseLayer

class Pooling(BaseLayer):
	
	def __init__(self,stride_shape, pooling_shape):
		BaseLayer.__init__(self)
		self.stride_shape = stride_shape
		self.pooling_shape = pooling_shape
	def forward(self, input_tensor):
		b, c, y, x = input_tensor.shape # input size
		ky, kx = self.pooling_shape # Kernel size (along height and width)
		sy, sx = self.stride_shape # strides along height and width
		outy = (y-ky)//sy + 1 # output height
		outx = (x-kx)//sx + 1 # output width
		self.input_tensor = input_tensor.copy()
		output = np.zeros((b, c, outy, outx))
		output_idx = np.zeros((b, c, outy, outx, ky, kx))
		strides = (sy*x, sx, x, 1) 
		strides = tuple(i * input_tensor[0,0].itemsize for i in strides)
		for batch in range(b):
			for channel in range(c):
				subM = np.lib.stride_tricks.as_strided(input_tensor[batch,channel], shape=(outy, outx, ky, kx), strides=strides)
				output[batch,channel] = np.max(subM, axis=(2,3))
				for idx_i, i in enumerate(subM):
					for idx_j, j in enumerate(i):
						output_idx[batch, channel,idx_i, idx_j] = 1 * (j == j.max(axis=(0,1))) #* output[batch,channel, idx_i, idx_j]
		self.maxlocs = output_idx
		self.forward_output = output
		self.subM_shape = (outy, outx, ky, kx)
		return output

	def backward(self, error_tensor):
		maxlocs = self.maxlocs.copy()
		d_tensor = np.zeros(self.input_tensor.shape)
		for batch in range(error_tensor.shape[0]):
			for channel in range(error_tensor.shape[1]):
				for y in range(error_tensor.shape[2]):
					for x in range(error_tensor.shape[3]):
						maxlocs[batch, channel, y, x] = np.multiply(maxlocs[batch, channel, y, x], error_tensor[batch, channel, y, x])
						d_tensor[batch, channel, (y*self.stride_shape[0]): min(y*self.stride_shape[0]+self.pooling_shape[0], self.input_tensor.shape[2]), (x*self.stride_shape[1]): min(x*self.stride_shape[1]+self.pooling_shape[1], self.input_tensor.shape[3])] += maxlocs[batch, channel, y, x]#subM[y, x]
		return d_tensor