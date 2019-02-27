import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import feature, color


# Generate noisy image of a square
import matplotlib.image as mpimg
im = mpimg.imread('small/Places365_val_00000001.jpg')
im = color.rgb2grey(im)

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(im, sigma=2.2)

# display results
plt.imshow(edges1, cmap=plt.cm.gray)
plt.show()