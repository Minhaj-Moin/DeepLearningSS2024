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
		self.input_tensor = input_tensor.copy()
		outy = (y-ky)//sy + 1 # output height
		outx = (x-kx)//sx + 1 # output width
		output = np.zeros((b, c, outy, outx))
		output_idx = np.zeros((b, c, outy, outx, ky, kx))
		output_idx_org = np.zeros((b, c, outy, outx, ky, kx))
		output_org2 = np.zeros(input_tensor.shape)
		strides = (sy*x, sx, x, 1) 
		strides = tuple(i * input_tensor[0,0].itemsize for i in strides)
		self.strides = strides
		# print("Strides", strides, input_tensor.strides)
		for batch in range(b):
			for channel in range(c):
				subM = np.lib.stride_tricks.as_strided(input_tensor[batch,channel], shape=(outy, outx, ky, kx), strides=strides)
				
				subM_loc = subM.copy()
				output[batch,channel] = np.max(subM, axis=(2,3))
				for idx_i, i in enumerate(subM):
					for idx_j, j in enumerate(i):
						subM_loc[idx_i, idx_j] = 1 * (j == j.max(axis=(0,1))) #* output[batch,channel, idx_i, idx_j]
				output_idx[batch, channel] = subM_loc
				# output_idx_org[batch, channel] = subM.copy()
				# print("DIFF",input_tensor.shape,subM.base, subM_loc.base, output_idx_org[batch, channel].strides, subM.shape, output_idx_org[batch, channel].shape, subM.itemsize, 
				# 	output_idx_org[batch, channel].itemsize)
				# output_org2[batch, channel] = np.lib.stride_tricks.as_strided(output_idx_org[batch, channel], shape=input_tensor[batch,channel].shape, strides=input_tensor[batch,channel].strides)
		self.maxlocs = output_idx
		# self.maxlocs_org = output_idx_org
		# self.output_org2 = output_org2#np.zeros(input_tensor.shape)
		self.forward_output = output
		self.subM_shape = (outy, outx, ky, kx)
		return output

	def backward(self, error_tensor):
		np.set_printoptions(linewidth=np.inf)
		maxlocs = self.maxlocs.copy()
		d_tensor = np.zeros(self.input_tensor.shape)
		i_tensor = self.input_tensor.copy()
		print("strides", np.array(self.strides)/self.input_tensor.itemsize)
		# print("ItemSize",self.input_tensor.itemsize, self.output_org2.itemsize, maxlocs.itemsize, self.maxlocs_org.itemsize)
		for batch in range(error_tensor.shape[0]):
			for channel in range(error_tensor.shape[1]):
				subM = np.lib.stride_tricks.as_strided(i_tensor[batch,channel], shape=self.subM_shape, strides=self.strides, subok=True, writeable=True)
				for idx_i, i in enumerate(subM):
					for idx_j, j in enumerate(i):
						subM[idx_i, idx_j] = (j == j.max(axis=(0,1))) * 1
				print("SubMShape", batch, channel, subM.shape, i_tensor[batch, channel].shape, error_tensor[batch, channel].shape, self.stride_shape)
				# for y in range(error_tensor.shape[2]):
				# 	for x in range(error_tensor.shape[3]):
				# 		# print("Back :", batch, channel, y ,x, maxlocs.shape, error_tensor.shape)
				# 		maxlocs[batch, channel, y, x] = np.multiply(maxlocs[batch, channel, y, x], error_tensor[batch, channel, y, x])
				# 		maxlocs[batch, channel, y, x][maxlocs[batch, channel, y, x]==0] = 1000
				for y in range(error_tensor.shape[2]):
					for x in range(error_tensor.shape[3]):
						# print(y, y+self.pooling_shape[0], x, x+self.pooling_shape[1])
						# print("Back :", batch, channel, y ,x, maxlocs.shape, error_tensor.shape)
						subM[y, x] = np.multiply(subM[y, x], error_tensor[batch, channel, y, x])
						maxlocs[batch, channel, y, x] = np.multiply(maxlocs[batch, channel, y, x], error_tensor[batch, channel, y, x])
						# print("||",error_tensor.shape, subM.shape, y, x, (y*self.stride_shape[0]), y+self.pooling_shape[0], (x*self.stride_shape[1]), x+self.pooling_shape[1])
						print(y, x, (y*self.stride_shape[0]), min(y*self.stride_shape[0]+self.pooling_shape[0], self.input_tensor.shape[2]), (x*self.stride_shape[1]), min(x*self.stride_shape[1]+self.pooling_shape[1], self.input_tensor.shape[3]),
							d_tensor[batch, channel, (y*self.stride_shape[0]): min(y*self.stride_shape[0]+self.pooling_shape[0], self.input_tensor.shape[2]), (x*self.stride_shape[1]): min(x*self.stride_shape[1]+self.pooling_shape[1], self.input_tensor.shape[3])].shape,
						 subM[y, x].shape, self.stride_shape)
						d_tensor[batch, channel, (y*self.stride_shape[0]): min(y*self.stride_shape[0]+self.pooling_shape[0], self.input_tensor.shape[2]), (x*self.stride_shape[1]): min(x*self.stride_shape[1]+self.pooling_shape[1], self.input_tensor.shape[3])] += subM[y, x]
						# maxlocs[batch, channel, y, x][maxlocs[batch, channel, y, x]==0] = 1000
					# print(np.concatenate(subM[y],axis=1))
					# z = np.concatenate(subM[y],axis=1)
					# d_tensor[batch, channel, :z.shape[0],:z.shape[1]] = z
				# d_tensor[batch, channel] = np.lib.stride_tricks.as_strided(subM, self.input_tensor[batch, channel].shape, self.input_tensor[batch, channel].strides).copy()
				# self.output_org2[batch, channel] = np.lib.stride_tricks.as_strided(self.maxlocs_org[batch, channel], self.input_tensor[batch, channel].shape, self.input_tensor[batch, channel].strides)
				# print("B",batch, channel, maxlocs[batch, channel].shape, np.array(self.input_tensor[batch, channel].strides), np.array(self.strides)/8, self.input_tensor.shape, self.input_tensor.itemsize)
				# print("batch", batch, channel,'\n', np.lib.stride_tricks.as_strided(maxlocs[batch, channel], self.input_tensor[batch, channel].shape, self.input_tensor[batch, channel].strides),
				# 	  '\nMaxLocs:\n', maxlocs[batch, channel, y, x])
		with open('f.txt', 'w') as outfile:
			# outfile.write(f"""error_tensor:\n{error_tensor.round(2)}\n {error_tensor.shape}\n
			# 	forward_tensor:\n{self.forward_output.round(2)}\n {self.forward_output.shape}\n
			# 	d_tensor:\n{d_tensor.round(2)}\n {d_tensor.shape}\n
			# 	MaxLocs:\n{maxlocs} \n{maxlocs.shape}\n
			# 	maxlocs_org:\n{self.maxlocs_org}\n{self.maxlocs_org.shape}\n
			# 	input_tensor:\n{self.input_tensor}\n {self.input_tensor.shape}\n
			# 	output_org2:\n{self.output_org2}\n {self.output_org2.shape}""")
			outfile.write(f"""
				input_tensor:\n{self.input_tensor}\n {self.input_tensor.shape}\n
				d_tensor:\n{d_tensor.round(2)}\n {d_tensor.shape}\n
				error_tensor:\n{error_tensor.round(2)}\n {error_tensor.shape}\n
				MaxLocs:\n{maxlocs} \n{maxlocs.shape}\n""")
		# with open('f2.txt', 'w') as outfile:
		# 	outfile.write(f"""
		# 		input_tensor:\n{self.input_tensor}\n {self.input_tensor.shape}\n
		# 		output_org2:\n{self.output_org2}\n {self.output_org2.shape}""")
		return d_tensor