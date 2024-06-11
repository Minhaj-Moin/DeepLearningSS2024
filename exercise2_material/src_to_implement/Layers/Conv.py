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

num_strides = 1



if type(num_strides) == int:
	num_strides = tuple((num_strides * np.ones(len(a.shape))).astype(int))
	print(num_strides, a.shape)


sub_shape = (3,3)
view_shape = tuple(np.subtract(a.shape, sub_shape) + 2 - (num_strides)) + sub_shape
print("view_shape",view_shape)
strides = a.strides + a.strides
strides = np.array(strides)
for k in range(len(a.strides)):
	strides[k] *= num_strides[k]

sub_matrices = np.lib.stride_tricks.as_strided(a,view_shape,strides)
conv_filter = np.array([[0,-1,0],
						[-1,5,-1],
						[0,-1,0]])
# conv_filter = np.array([[6,7,8],
# 						[11,12,13],
# 						[16,17,18]])
m = np.einsum('ij,ijkl->kl',conv_filter,sub_matrices)
# print(sub_matrices)
print(m)
from scipy import signal
corr = signal.correlate2d(a, conv_filter, boundary='fill', mode='valid')
print((corr))

def conv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = np.zeros((img.shape))
    #Looping through the image to apply the convolution operation.
    for r in np.uint16(np.arange(filter_size/2.0, 
                          img.shape[0]-filter_size/2.0+1)):
        for c in np.uint16(np.arange(filter_size/2.0, 
                                           img.shape[1]-filter_size/2.0+1)):
            """
            Getting the current region to get multiplied with the filter.
            How to loop through the image and get the region based on 
            the image and filer sizes is the most tricky part of convolution.
            """
            curr_region = img[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)), 
                              c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0))]
            #Element-wise multipliplication between the current region and the filter.
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result) #Summing the result of multiplication.
            result[r, c] = conv_sum #Saving the summation in the convolution layer feature map.
            
    #Clipping the outliers of the result matrix.
    final_result = result[np.uint16(filter_size/2.0):result.shape[0]-np.uint16(filter_size/2.0), 
                          np.uint16(filter_size/2.0):result.shape[1]-np.uint16(filter_size/2.0)]
    return final_result
print(conv_(a,conv_filter))
# import matplotlib.pyplot as plt
# plt.imshow(corr)
# plt.show()