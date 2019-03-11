from __future__ import print_function, division
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from skimage import feature, color
import numpy as np
import random

import tarfile
import io
import pandas as pd

from torch.utils.data import Dataset


# from utils.Halftoning.halftone import generate_halftone


class PlacesDataset(Dataset):
    def __init__(self, txt_path='data/filelist.txt', img_dir='data.tar', transform=None):
        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_dir: path to image files as a uncompressed tar archive
        :param txt_path: a text file containing names of all of images line by line
        :param transform: apply some transforms like cropping, rotating, etc on input image
        :return: a 3-value dict containing input image (y_descreen) as ground truth, input image X as halftone image and edge-map (y_edge) of ground truth image to feed into the network.
        """

        df = pd.read_csv(txt_path, sep=' ', index_col=0)
        self.img_names = df.index.values
        self.txt_path = txt_path
        self.img_dir = img_dir
        self.transform = transform
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()

    def get_image_by_name(self, name):
        """
        Gets a image by a name gathered from file list csv file

        :param name: name of targeted image
        :return: a PIL image
        """
        with tarfile.open(self.img_dir) as tf:
            tarinfo = tf.getmember(name)
            image = tf.extractfile(tarinfo)
            image = image.read()
            image = Image.open(io.BytesIO(image))
        return image

    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        return len(self.img_names)

    def __getitem__(self, index):
        """
        Generate one item of data set. Here we apply our preprocessing things like halftone styles and
        subtractive color process using CMYK color model, generating edge-maps, etc.

        :param index: index of item in IDs list

        :return: a sample of data as a dict
        """

        X = self.get_image_by_name(self.img_names[index])

        if self.transform is not None:
            X = self.transform(X)

        # generate edge-map
        y_edge = self.canny_edge_detector(X)
        y_edge = self.to_tensor(y_edge)

        sample = {'X': X,
                  'y_edge': y_edge}

        return sample

    def canny_edge_detector(self, image):
        """
        Returns a binary image with same size of source image which each pixel determines belonging to an edge or not.

        :param image: PIL image
        :return: Binary numpy array
        """
        image = self.to_pil(image)
        image = image.convert(mode='L')
        image = np.array(image)
        edges = feature.canny(image, sigma=1)  # TODO: the sigma hyper parameter value is not defined in the paper.
        size = edges.shape[::-1]
        data_bytes = np.packbits(edges, axis=1)
        edges = Image.frombytes(mode='1', size=size, data=data_bytes)
        return edges


# https://discuss.pytorch.org/t/adding-gaussion-noise-in-cifar10-dataset/961/2
class RandomNoise(object):
    def __init__(self, p, mean=0, std=1):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if random.random() <= self.p:
            return img.clone().normal_(self.mean, self.std)
        return img


# %% test
def canny_edge_detector(image):
    """
    Returns a binary image with same size of source image which each pixel determines belonging to an edge or not.

    :param image: PIL image
    :return: Binary numpy array
    """

    image = image.convert(mode='L')
    image = np.array(image)
    edges = feature.canny(image, sigma=1)  # TODO: the sigma hyper parameter value is not defined in the paper.
    size = edges.shape[::-1]
    databytes = np.packbits(edges, axis=1)
    edges = Image.frombytes(mode='1', size=size, data=databytes)
    # https://gist.github.com/PM2Ring/b09216123cca86e9b9cf889bfd3c5cec
    return edges


def get_image_by_name(img_dir, name):
    """
    gets a image by a name gathered from file list csv file

    :param img_dir: Directory to image files as a uncompressed tar archive
    :param name: name of targeted image
    :return: a PIL image
    """

    with tarfile.open(img_dir) as tf:
        tarinfo = tf.getmember(name)
        image = tf.extractfile(tarinfo)
        image = image.read()
        image = Image.open(io.BytesIO(image))
    return image


# %% test 2
# z = get_image_by_name('data/data.tar', 'Places365_val_00000002.jpg')
# ze = canny_edge_detector(z)
# ze.show()