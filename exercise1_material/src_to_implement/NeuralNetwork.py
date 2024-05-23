import numpy as np
if __name__=='__main__':
	from Layers import *
	from Optimization import *
	import matplotlib.pyplot as plt
	import tabulate
	import argparse
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
			self.backward()
	def test(self,input_tensor):
		return self.forward(input_tensor)

if __name__=='__main__':
	net = NeuralNetwork(Optimizers.Sgd(1e-3))
	categories = 3
	input_size = 4
	net.data_layer = Helpers.IrisData(50)
	net.loss_layer = Loss.CrossEntropyLoss()

	fcl_1 = FullyConnected.FullyConnected(input_size, categories)
	net.append_layer(fcl_1)
	net.append_layer(ReLU.ReLU())
	fcl_2 = FullyConnected.FullyConnected(categories, categories)
	net.append_layer(fcl_2)
	net.append_layer(SoftMax.SoftMax())

	print(net.layers[0].weights)
	net.train(1)
	print(net.layers[0].weights)
	plt.figure('Loss function for a Neural Net on the Iris dataset using SGD')
	plt.plot(net.loss, '-x')
	plt.show()
	data, labels = net.data_layer.get_test_set()
	results = net.test(data)
	index_maximum = np.argmax(results, axis=1)
	one_hot_vector = np.zeros_like(results)
	for i in range(one_hot_vector.shape[0]):
		one_hot_vector[i, index_maximum[i]] = 1

	correct = 0.
	wrong = 0.
	for column_results, column_labels in zip(one_hot_vector, labels):
		if column_results[column_labels > 0].all() > 0:
			correct += 1
		else:
			wrong += 1

	accuracy = correct / (correct + wrong)
	print('\nOn the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%')
