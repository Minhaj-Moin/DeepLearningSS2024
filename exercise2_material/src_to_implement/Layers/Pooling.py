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
		output = np.zeros((b, c, outy, outx))
		output_idx = np.zeros((b, c, outy, outx))

		for batch in range(b):
			for channel in range(c):
				strides = (sy*x, sx, x, 1) 
				strides = tuple(i * input_tensor[batch,channel].itemsize for i in strides) 
				subM = np.lib.stride_tricks.as_strided(input_tensor[batch,channel], shape=(outy, outx, ky, kx), strides=strides)
				output[batch,channel] = np.max(subM, axis=(2,3))
				output_idx[batch,channel] = (1.0 * (subM[:,:,0,0] == subM.max(axis=(2,3))))
		self.maxlocs = output_idx
		print(output_idx)
		return output

	def backward(self, error_tensor):
		pass