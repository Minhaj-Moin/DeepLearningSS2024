import os.path
import os
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path: str, label_path: str, batch_size: int, image_size: int, rotation=False, mirroring=False , shuffle=False ):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        self.file_path = file_path
        self.files = np.array(os.listdir(file_path))
        self.files.sort()
        self.labels = json.load(open(label_path,'r'))
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.cursor = 0
        self.roll = 0
        self.epoch = 0

        ## Shuffle the list for the first time.
        if self.shuffle: self.files = np.random.permutation(self.files)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        
        ## If cursor nears the end of dataset, append some images to the list so we can have evenly sized batches.
        if (self.cursor+self.batch_size) > len(self.files):
            self.roll = (self.batch_size)
            self.epoch += 1
            ## Shuffle the dataset at the new epoch
            if self.shuffle: self.files = np.random.permutation(self.files)

        ## self.roll is a variable, at near the end it will become equal to the batch size, or else it would be 0 so we will append an empty array
        lst = np.append(self.files,self.files[:self.roll])[max(0,self.cursor):self.cursor+self.batch_size]
        if self.shuffle:
            lst = np.random.permutation(lst)
        self.cursor+= self.batch_size
        
        ## Loading is only done when the next() function is called to make it lazy.

        return np.array([self.augment(resize(np.load(self.file_path+i), self.image_size)) for i in lst]), [(self.labels[k.replace('.npy','')]) for k in lst]
        # ImgLoad = np.vectorize(lambda x: self.augment(resize(np.load(self.file_path+str(x)), self.image_size)))
        # Label = np.vectorize(lambda x: self.labels[x.replace('.npy','')])
        # return ImgLoad(np.array(lst)), LabelLoad(lst)
        #return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        Img = img.copy()
        if self.mirroring: Img = np.flip(Img, axis=np.random.randint(0,2))
        if self.rotation: Img = np.rot90(Img, k=np.random.randint(0,4))
        return Img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[x]
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        
        x = self.next()

        ## for 60 batch size it will be 6 x 10 subplots
        fig, axes = plt.subplots(max(1,self.batch_size//10), min(10,self.batch_size), figsize = (50,50))
        # angle = np.random.randint(0,4)
        # plt.imshow(np.rot90(x[0][0], angle))
        # print(axes)
        for ax, img,label in zip(axes.flatten(), x[0], x[1]):
            ax.set_axis_off()
            ax.imshow(img)
            ax.set_title(self.class_name(label))
        plt.show()

