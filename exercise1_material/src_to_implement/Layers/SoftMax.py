from  Layers.Base  import BaseLayer
import numpy as np
class SoftMax(BaseLayer):
	
	def __init__(self):
		BaseLayer.__init__(self)
		self.value = None

	def forward(self, input_tensor):
		exps = np.exp(input_tensor - np.max(input_tensor)) 
		self.value = exps / exps.sum()
		return self.value
		
	def backward(self, error_tensor):
		idx = np.arange(min(error_tensor.shape[0],error_tensor.shape[1]))
		x = error_tensor.copy()
		x = -np.dot(x,x)
		x[idx,idx] = x(1-x)
		return x
		
	