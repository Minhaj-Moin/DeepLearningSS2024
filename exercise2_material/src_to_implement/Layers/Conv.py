import numpy as np
if __name__=='__main__':
	from Base import BaseLayer
	from Initializers import UniformRandom
else: 
	from Layers.Base import BaseLayer
	from Layers.Initializers import UniformRandom
import unittest

class Conv(BaseLayer):
	def __init__(self, stride_shape, convolution_shape, num_kernels):
		BaseLayer.__init__(self)
		self.trainable=True
		self.weights = None
		self.bias = None
		self.stride_shape = stride_shape
		self.convolution_shape = convolution_shape
		self.num_kernels = num_kernels
		self.initialize(UniformRandom(),UniformRandom())

	def initialize(self, weights_initializer, bias_initializer):
		# print("initialize CAlled")
		self.weights = weights_initializer.initialize(((self.num_kernels,)+ self.convolution_shape), np.prod(self.convolution_shape), np.prod(self.convolution_shape[1:]) * self.num_kernels)
		self.bias = weights_initializer.initialize((self.num_kernels,1), np.prod(self.convolution_shape), np.prod(self.convolution_shape[1:]) * self.num_kernels)

	def forward(self, input_tensor):
		'''input_tensor shape (b, c, y) or (b, c, y, x)'''
		from scipy import signal
		from scipy.ndimage import correlate1d
		outshape = [input_tensor.shape[0], self.num_kernels, input_tensor.shape[2]]
		if len(input_tensor.shape)>3: outshape += [input_tensor.shape[3]]
		output = np.zeros(outshape)
		for b in range(input_tensor.shape[0]):
			for ch in range(input_tensor.shape[1]):
				for ker in range(self.num_kernels):
					if len(self.convolution_shape) == 2:
						print("conv shape",self.convolution_shape)
						print("INPUTSHAPES",input_tensor.shape, input_tensor[b, ch].shape, self.weights[ker,ch].shape, self.weights.shape, outshape, self.convolution_shape)
						output[b,ker] = correlate1d(input_tensor[b,ch], weights=self.weights[ker,ch], mode='constant') + self.bias[ker]
					elif len(self.convolution_shape) == 3:
						# print("conv shape",self.convolution_shape)
						# print(input_tensor.shape, input_tensor[b, ch].shape, self.weights[ker,ch].shape, self.weights.shape)
						output[b,ker] = signal.correlate2d(input_tensor[b,ch], self.weights[ker,ch], boundary='fill', mode='same') + self.bias[ker]
		self.input_tensor = input_tensor
		## Striding
		if len(self.convolution_shape) == 2: return output[:, :, ::self.stride_shape[0]]
		if len(self.convolution_shape) == 3: return output[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]

	def backward(self, error_tensor):
		print(error_tensor.shape, self.weights.shape, self.num_kernels)
		a = self.conv_backward_naive(error_tensor)
		self.gradient_weights = a[1]
		self.gradient_bias = a[2]
		return a[0]
		return a
	def conv_backward_naive(self, dout):
		"""
		A naive implementation of the backward pass for a convolutional layer.

		Inputs:
		- dout: Upstream derivatives.
		- cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

		Returns a tuple of:
		- dx: Gradient with respect to x
		- dw: Gradient with respect to w
		- db: Gradient with respect to b
		"""
		
		dx, dw, db = None, None, None

		x = self.input_tensor
		w = self.weights
		b = self.bias
		pad = 0#conv_param['pad']
		stride = 0#conv_param['stride']
		# dout = error_tensor

		# Initialisations
		dx = np.zeros_like(self.input_tensor)
		dw = np.zeros_like(self.weights)
		db = np.zeros_like(self.bias)
		
		# Dimensions
		N, C, H, W = x.shape
		F, _, HH, WW = w.shape
		_, _, H_, W_ = dout.shape
		
		# db - dout (N, F, H', W')
		# On somme sur tous les éléments sauf les indices des filtres
		db = np.sum(dout, axis=(0, 2, 3))
		
		# dw = xp * dy
		# 0-padding juste sur les deux dernières dimensions de x
		xp = np.pad(x, ((0,), (0,), (pad,), (pad, )), 'constant')
		
		# Version sans vectorisation
		for n in range(N):	   # On parcourt toutes les images
			for f in range(F):   # On parcourt tous les filtres
				for i in range(HH): # indices du résultat
					for j in range(WW):
						for k in range(H_): # indices du filtre
							for l in range(W_):
								for c in range(C): # profondeur
									dw[f,c,i,j] += xp[n, c, stride*i+k, stride*j+l] * dout[n, f, k, l]

		# dx = dy_0 * w'
		# Valide seulement pour un stride = 1
		# 0-padding juste sur les deux dernières dimensions de dy = dout (N, F, H', W')
		doutp = np.pad(dout, ((0,), (0,), (WW-1,), (HH-1, )), 'constant')

		# 0-padding juste sur les deux dernières dimensions de dx
		dxp = np.pad(dx, ((0,), (0,), (pad,), (pad, )), 'constant')

		# filtre inversé dimension (F, C, HH, WW)
		w_ = np.zeros_like(w)
		for i in range(HH):
			for j in range(WW):
				w_[:,:,i,j] = w[:,:,HH-i-1,WW-j-1]
		
		# Version sans vectorisation
		for n in range(N):	   # On parcourt toutes les images
			for f in range(F):   # On parcourt tous les filtres
				for i in range(H+2*pad): # indices de l'entrée participant au résultat
					for j in range(W+2*pad):
						for k in range(HH): # indices du filtre
							for l in range(WW):
								for c in range(C): # profondeur
									dxp[n,c,i,j] += doutp[n, f, i+k, j+l] * w_[f, c, k, l]
		#Remove padding for dx
		# print("dxp",dxp.shape)
		dx = dxp#[:,:,pad:-pad,pad:-pad]

		return dx, dw, db

	def backward_pass(self, da_curr: np.array) -> np.array:
		"""
		:param da_curr - 4D tensor with shape (n, h_out, w_out, n_f)
		:output 4D tensor with shape (n, h_in, w_in, c)
		------------------------------------------------------------------------
		n - number of examples in batch
		w_in - width of input volume
		h_in - width of input volume
		w_out - width of input volume
		h_out - width of input volume
		c - number of channels of the input volume
		n_f - number of filters in filter volume
		"""
		_, _, h_out, w_out = da_curr.shape
		n, _, h_in, w_in = self.input_tensor.shape
		h_f, w_f, _, _ = self.weights.shape
		pad = 0 #self.calculate_pad_dims()
		a_prev_pad = np.pad(self.input_tensor, ((0,), (0,), (pad,), (pad, )), 'constant') #self.pad(array=self._a_prev, pad=pad)
		output = np.zeros_like(a_prev_pad)

		self.gradient_bias = da_curr.sum(axis=(0, 1, 2)) / n
		self.gradient_weights = np.zeros_like(self.weights)

		for i in range(h_out):
			for j in range(w_out):
				h_start = i * 1
				h_end = h_start + h_f
				w_start = j * 1
				w_end = w_start + w_f
				output[:, h_start:h_end, w_start:w_end, :] += np.sum(
					self.weights[np.newaxis, :, :, :, :] *
					da_curr[:, i:i+1, j:j+1, np.newaxis, :],
					axis=4
				)
				self._dw += np.sum(
					a_prev_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
					da_curr[:, i:i+1, j:j+1, np.newaxis, :],
					axis=0
				)

		self._dw /= n
		return output[:, pad[0]:pad[0]+h_in, pad[1]:pad[1]+w_in, :]


class TestConv(unittest.TestCase):
    plot = False
    directory = 'plots/'

    class TestInitializer:
        def __init__(self):
            self.fan_in = None
            self.fan_out = None

        def initialize(self, shape, fan_in, fan_out):
            self.fan_in = fan_in
            self.fan_out = fan_out
            weights = np.zeros((1, 3, 3, 3))
            weights[0, 1, 1, 1] = 1
            return weights

    def setUp(self):
        self.batch_size = 2
        self.input_shape = (3, 10, 14)
        self.input_size = 14 * 10 * 3
        self.uneven_input_shape = (3, 11, 15)
        self.uneven_input_size = 15 * 11 * 3
        self.spatial_input_shape = np.prod(self.input_shape[1:])
        self.kernel_shape = (3, 5, 8)
        self.num_kernels = 4
        self.hidden_channels = 3

        self.categories = 105
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    

    def test_1D_forward_size(self):
        conv = Conv([2], (3, 3), self.num_kernels)
        input_tensor = np.array(range(3 * 15 * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape((self.batch_size, 3, 15))
        output_tensor = conv.forward(input_tensor)
        print(output_tensor)
        self.assertEqual(output_tensor.shape,  (self.batch_size, self.num_kernels, 8),
                         "Possible reason: If any other tests for the forward pass fail, fix those first. Otherwise, 1D"
                         "correlation is not implemented correctly. Make sure to differentiate between the 1D and 2D"
                         "case in your forward pass.")
t = TestConv()
t.setUp()
t.test_1D_forward_size()