import numpy as np
class CrossEntropyLoss:
	def __init__(self):
		self.preds = None
	def forward(self,prediction_tensor, label_tensor,epsilon=np.finfo('float64').eps):
		self.preds = prediction_tensor
		return -np.sum(label_tensor*np.log(prediction_tensor+epsilon))#/prediction_tensor.shape[0]
	def backward(self,label_tensor,epsilon=np.finfo('float64').eps):
		return -label_tensor/(self.preds+epsilon)