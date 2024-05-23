if __name__=='__main__':
	from Base import BaseLayer
else: from Layers.Base import BaseLayer
import numpy as np
class SoftMax(BaseLayer):
	
	def __init__(self):
		BaseLayer.__init__(self)
		self.value = None

	def forward(self, input_tensor):
		
		exps = np.exp(input_tensor.T - np.max(input_tensor,axis=1)).T
		# print("INPUT",input_tensor,'\n', exps)
		self.value = (exps.T / np.sum(exps,axis=1)).T
		return self.value.round(7)
	


	def backward(self, error_tensor):
		# idx = np.arange(min(error_tensor.shape[0],error_tensor.shape[1]))
		# x = error_tensor.copy()
		# x = error_tensor * error_tensor
		# print(idx)
		# x[idx,idx] = x*(1-x)
		# grad = -np.outer(self.value, self.value) + np.diag(self.value.flatten())
		x = []
		for i in range(len(error_tensor)):
			x.append(np.dot(np.diagflat(error_tensor[i]) - np.dot(error_tensor[i], error_tensor[i].T),self.value[i]))
		return x#np.dot(np.diagflat(error_tensor) - np.dot(error_tensor, error_tensor.T),self.value)
		# print("WPW",(self.value * (error_tensor - (self.value * error_tensor).sum(axis=0))).shape, self.value.shape, error_tensor.shape)
		return self.value * (error_tensor - np.multiply(self.value,error_tensor).sum(axis=0)).round(3)
		
# input_tensor = self.label_tensor - 1.
# input_tensor *= -100.
layer = SoftMax()
pred = layer.forward((np.array([[100.,  -0. ,100. ,100.], [ -0., 100., 100., 100.], [100., 100., 100,  -0.], [100, 100, 100,  -0.], 
	[ -0., 100., 100., 100.], [100., 100., 100.,  -0.], [100., 100.,  -0., 100.], [100.,  -0., 100., 100.], [100.,  -0., 100., 100.]])-100)*-1)
print(pred)