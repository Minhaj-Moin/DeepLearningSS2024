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
		self.input_tensor = input_tensor
		outy = (y-ky)//sy + 1 # output height
		outx = (x-kx)//sx + 1 # output width
		output = np.zeros((b, c, outy, outx))
		output_idx = np.zeros((b, c, outy, outx, ky, kx))
		output_idx_org = np.zeros((b, c, outy, outx, ky, kx))
		output_idx_org = np.zeros((b, c, outy, outx, ky, kx))
		strides = (sy*x, sx, x, 1) 
		strides = tuple(i * input_tensor[0,0].itemsize for i in strides)
		self.strides = strides
		for batch in range(b):
			for channel in range(c):
				subM = np.lib.stride_tricks.as_strided(input_tensor[batch,channel], shape=(outy, outx, ky, kx), strides=strides)
				subM_loc = subM.copy()
				output[batch,channel] = np.max(subM, axis=(2,3))
				for idx_i, i in enumerate(subM):
					for idx_j, j in enumerate(i):
						subM_loc[idx_i, idx_j] = 1 * (j == j.max(axis=(0,1)))
				output_idx[batch, channel] = subM_loc
				output_idx_org[batch, channel] = subM
		self.maxlocs = output_idx
		self.maxlocs_org = output_idx_org
		self.forward_output = output
		return output

	def backward(self, error_tensor):
		maxlocs = self.maxlocs.copy()
		d_tensor = np.zeros(self.input_tensor.shape)
		for batch in range(error_tensor.shape[0]):
			for channel in range(error_tensor.shape[1]):
				for y in range(error_tensor.shape[2]):
					for x in range(error_tensor.shape[3]):

						maxlocs[batch, channel, y, x] = np.multiply(maxlocs[batch, channel, y, x], error_tensor[batch, channel, y, x])
				d_tensor[batch, channel] = np.lib.stride_tricks.as_strided(maxlocs[batch, channel], self.input_tensor[batch, channel].shape, self.input_tensor[batch, channel].strides)
				print("B",batch, channel, maxlocs[batch, channel].shape)
		with open('f.txt', 'w') as outfile:
			outfile.write(f"error_tensor:\n{error_tensor.round(2)}\n {error_tensor.shape}\nerror_tensor:\n{self.forward_output.round(2)}\n {self.forward_output.shape}\n d_tensor:\n{d_tensor.round(2)}\n {d_tensor.shape}\nMaxLocs\n{maxlocs} \n{maxlocs.shape}\nmaxlocs_org:\n{self.maxlocs_org}\n{self.maxlocs_org.shape}\ninput_tensor:\n{self.input_tensor}\n {self.input_tensor.shape}\n ")
		return d_tensor