class Sgd:
	def __init__(self, learning_rate):
		self.learning_rate = learning_rate

	def calculate_update(self,weight_tensor, gradient_tensor):
		return weight_tensor - self.learning_rate*gradient_tensor
	def copy(self, deep=True):
		if deep:
			return Sgd(self.learning_rate)
		return self
