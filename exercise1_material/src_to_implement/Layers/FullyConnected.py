import numpy as np
print(__name__)
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
	
	def __init__(self,input_size, output_size):
		BaseLayer.__init__(self)
		self.trainable = True
		self.weights = np.random.uniform(0,1,(input_size+1,output_size))
		self.input_size = input_size
		self.output_size = output_size

		self._optimizer = None
		self._gradient_weights = None
	@property
	def optimizer(self): return self._optimizer

	@optimizer.setter
	def optimizer(self, optimizer): self._optimizer = optimizer
	
	@property
	def gradient_weights(self): return self._gradient_weights

	@gradient_weights.setter
	def gradient_weights(self, gradient_weights):  self._gradient_weights = gradient_weights
	def forward(self, input_tensor):
		b = np.ones((input_tensor.shape[0],input_tensor.shape[1]+1))
		b[:,:-1] = input_tensor
		# print("InputTensor",input_tensor,b, 'END')
		# print("Weights, inputSize, InputTensor",self.weights.shape, self.input_size, input_tensor.shape )
		return np.dot(b,self.weights)
	def backward(self, error_tensor):
		# print(error_tensor.shape, self.weights.shape, self.input_size )
		# self.weights = np.dot(error_tensor,self.weights.transpose()).T#[:,:-1]
		self._gradient_weights = np.dot(error_tensor,self.weights.transpose())[:,:-1]
		return np.dot(error_tensor,self.weights.transpose())[:,:-1]#self.weights
	def calculate_update(self,weight_tensor, gradient_tensor):
		if self._optimizer: return True
		pass
