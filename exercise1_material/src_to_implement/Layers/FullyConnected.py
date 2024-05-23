import numpy as np
print(__name__)
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
	
	def __init__(self,input_size, output_size):
		BaseLayer.__init__(self)
		self.trainable = True
		self.weights = np.random.uniform(0,1,(input_size+1,output_size))
		self.bias = np.zeros(output_size)
		self.input_size = input_size
		self.output_size = output_size
		self.input_tensor = None

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
		self.input_tensor = b.copy()
		# print("InputTensor",input_tensor,b, 'END')
		# print("Weights, inputSize, InputTensor",self.weights.shape, self.input_size, input_tensor.shape )
		return np.dot(b,self.weights)
	def backward(self, error_tensor):
		# print(error_tensor.shape, self.weights.shape, self.input_size )
		
		self._gradient_weights = np.clip(np.dot(error_tensor,self.weights.transpose())[:,:-1],-1,1)
		self.weights = self.calculate_update(self.weights, self._gradient_weights)

		return np.dot(error_tensor,self.weights.transpose())[:,:-1]#self.weights
	def calculate_update(self,weight_tensor, gradient_tensor):
		if self._optimizer: 
			print('YOLO=',weight_tensor.shape, self.input_tensor.T.shape, gradient_tensor.shape)
			weight_tensor -= self._optimizer.learning_rate * np.dot(self.input_tensor.T,gradient_tensor)[:,:-1]
		return weight_tensor
