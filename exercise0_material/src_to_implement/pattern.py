import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution: int, tile_size: int):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.zeros((self.resolution,self.resolution))

    def draw(self):

        ## resolution should be divisible by tile size * 2, or else we get an empty image.
        ## Only create a pattern, with np arrays, then we repeat the pattern in a square shape 
        ## to get to the desired resolution and tile size.

        if self.resolution % (2*self.tile_size) == 0:
            check = np.zeros((self.resolution//self.tile_size,self.resolution//self.tile_size))
            check[::2,1::2] = 1 ## Even rows, odd columns white
            check[1::2, ::2] = 1 ## Odd rows, even columns white
            ## Repeat the pattern so instead of a pixel we have a square of pixels
            self.output = np.repeat(np.repeat(check,self.tile_size,axis=1),self.tile_size,axis=0)
            return self.output.copy()
        return self.output.copy()
            
    def show(self):
        if self.output is not None:
            plt.axis('off')
            plt.imshow(self.output, cmap='gray')
            plt.show()
        else:
            return None
        
class Circle:
    def __init__(self, resolution:int, radius:int, position:tuple):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = np.zeros((self.resolution,self.resolution))
    def draw(self):
        ## in numpy, adding two 2d arrays with dimensions like (1,x) and (y,1) will result in a 2d array with (x,y) shape
        ## [[1,2,3]] + [[1],[2],[3]] = [[2,3,4],[3,4,5],[4,5,6]]
        ## we can then use this with the circle equation to form a grid of values that will be increasing squared distances
        ## from the circle centre point, we can select those cells that are lower than radius**2 and set them to 1
        rr = np.arange(self.resolution)
        circle_eq = (rr[np.newaxis,:]-self.position[0])**2 + (rr[:,np.newaxis]-self.position[1])**2 < self.radius**2
        self.output[circle_eq] = 1
        return self.output.copy()
        
    def show(self):
        if self.output is not None:
            plt.axis('off')
            plt.imshow(self.output, cmap='gray')
            plt.show()
        else:
            return None

class Spectrum:
    def __init__(self, resolution: int):
        self.resolution = resolution
        self.output = np.zeros((self.resolution,self.resolution,3))

    def draw(self):

        ## Looking at the spectrum image, it was obvious we are getting increasing
        ## values of blue from the left, increasing values for red from the right,
        ## and increasing values for green from top to bottom

        self.output[:,:,0] = np.arange(0,self.resolution)/self.resolution
        self.output[:,:,1] = np.arange(0,self.resolution)/self.resolution
        self.output[:,:,1] = self.output[:,:,1].transpose()
        self.output[:,:,2] = np.arange(self.resolution,0,-1)/self.resolution
        return self.output.copy()
        
    def show(self):
        if self.output is not None:
            plt.axis('off')
            plt.imshow(self.output)
            plt.show()
        else:
            return None