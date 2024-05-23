import numpy as np
if __name__=='__main__':
	from Base import BaseLayer
else: from Layers.Base import BaseLayer
class NeuralNetwork():
	"""docstring for NeuralNetwork"""
	def __init__(self, optimizer):
		self.optimizer = optimizer
		self.loss = []
		self.layers = []
		self.loss_layer = []
		self.data_layer = None
	def forward(self, input_tensor_ = None):
		if input_tensor_ is None:
			self.input_tensor, self.label_tensor = self.data_layer.next()
			input_tensor_ = self.input_tensor.copy()
			for i in self.layers: input_tensor_ = i.forward(input_tensor_)
			return self.loss_layer.forward(input_tensor_, self.label_tensor)
		for i in self.layers: input_tensor_ = i.forward(input_tensor_)
		return input_tensor_
	def backward(self):
		error_tensor = self.loss_layer.backward(self.label_tensor)
		for k in self.layers[::-1]: error_tensor = k.backward(error_tensor)
	def append_layer(self,layer):
		if layer.trainable: layer.optimizer = self.optimizer.copy(deep=True)
		self.layers.append(layer)
	def train(self,iterations):
		for i in range(iterations):
			self.loss.append(self.forward())
	def test(self,input_tensor):
		return self.forward(input_tensor)