import numpy as np
# class Conv(BaseLayer):
# 	def __init__(self, stride_shape, convolution_shape, num_kernels):
# 		BaseLayer.__init__(self)
# 		self.trainable=True
# 		self.weights = None
# 		self.bias = None
# 		self.stride_shape = stride_shape
# 		self.convolution_shape = convolution_shape
# 		self.num_kernels = num_kernels

# 	def forward(self, input_tensor):
# 		'''shape (b, c, y) or (b, c, y, x)'''
# 		def conv2d(a, f):
# 		    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
# 		    subM = np.lib.stride_tricks.as_strided(a, shape = s, strides = a.strides * 2)
# 		    return np.einsum('ij,ijkl->kl', f, subM)
import numpy as np

a = np.array([ [ 0,  1,  2,  3,  4],
    		   [ 5,  6,  7,  8,  9],
    		   [10, 11, 12, 13, 14],
    		   [15, 16, 17, 18, 19],
    		   [20, 21, 22, 23, 24]])

num_strides = 2
sub_shape = (3,3)
W = a.shape[0]
K = sub_shape[0]
P = 1
S = num_strides
output_shape = np.floor((W-K+2*P)/S)+1
a = np.pad(a, (P, P))
print("output:", output_shape)
view_shape = tuple(np.subtract(a.shape, sub_shape) + 2 - num_strides) + sub_shape
print(view_shape)
strides = a.strides  * 2
strides = np.array(strides)
strides[0] *= num_strides
strides[1] *= num_strides
print(strides, np.divide(strides,4))

sub_matrices = np.lib.stride_tricks.as_strided(a,view_shape,strides)
print(sub_matrices, a.strides)
