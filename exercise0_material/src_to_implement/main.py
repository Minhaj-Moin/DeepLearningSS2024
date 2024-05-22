import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from pattern import Checker,Circle,Spectrum
from generator import ImageGenerator

print("Checker(250,25)")

s = Checker(250,25)
s.draw()
s.show()

input("Press Enter to continue")

print("Circle(1000,100,(500,500))")
s = Circle(1000,100,(500,500))
s.draw()
s.show()

input("Press Enter to continue")

print("Spectrum(250)")
s = Spectrum(250)
s.draw()
s.show()

input("Press Enter to continue")

print("ImageGenerator with 60 Images, no augmentation")
gen = ImageGenerator('./exercise_data/', './Labels.json',60, [32, 32, 3], rotation=False, mirroring=False,
                     shuffle=False)

gen.show()
input("Press Enter to continue")

print("ImageGenerator with 60 Images, rotation and mirroring enabled")
gen = ImageGenerator('./exercise_data/', './Labels.json',60, [32, 32, 3], rotation=True, mirroring=True,
                     shuffle=False)

gen.show()