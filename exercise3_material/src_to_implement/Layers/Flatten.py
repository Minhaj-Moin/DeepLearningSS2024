import numpy as np
if __name__=='__main__':
    from Base import BaseLayer
    import unittest
else: from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self) -> None:
        BaseLayer.__init__(self)
        self.input_shape = None
        pass

    def forward(self, input_tensor):
        # new shape is batch size x flattened tensor
        self.input_shape = input_tensor.shape
        total_elements = np.prod(self.input_shape)
        return input_tensor.reshape(
            (self.input_shape[0], total_elements // self.input_shape[0])
        )

    def backward(self, error_tensor):

        return error_tensor.reshape(self.input_shape)
