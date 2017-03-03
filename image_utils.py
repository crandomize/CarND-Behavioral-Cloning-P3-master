"""
Helpers for image handling during the preprocessing phases
"""

import numpy as np
import cv2


def driveImagePipeline(image):
    """
    Steps to convert image from the Drive process to model
    """
    #return cropAndResize(PILimage2CV2HSV(image))

    #return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
    #return cropAndResize(cv2.cvtColor(i,cv2.COLOR_BGR2YUV))
    return cropAndResize(np.array(image))

def trainingImagePipeline(imagefile):
    """
    Steps to convert image from the original image to training 
    """   
    #img = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
    #img = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2YUV)
    return cropAndResize(img)

def cropAndResize(image):
    """
    Crop image to dims and then resize
    64 rows 96 columns:  so x:96 y:64 
    """
    return cv2.resize(image[50:130], (96, 64), cv2.INTER_AREA)
    
def imgload2HSV(imagefile):
    """
    It loads an image from file system, returning a CV2 image in HSV space
    Args:
        imagefile: file path of the image
    Returns:
        image: np.ndarray  (as CV2 images use numpy arrays)
    """
    img = cv2.imread(imagefile)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def imgload(imagefile):
    """
    It loads an image from file system, returning a CV2 image in RGB space
    Args:
        imagefile: file path of the image
    Returns:
        image: np.ndarray  (as CV2 images use numpy arrays)
    """
    img = cv2.imread(imagefile)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def getflipped(image):
    """
    Flips the image,  rotating it around vertical line
    Args:
        image (cv2 ndarray)
    Returns:
        image flipped (cv2 ndarray)
    """
    newimg = image.copy()
    newimg = cv2.flip(newimg, 1)
    return newimg

def PILimage2CV2HSV(image):
    """
    Input image if original in RGB space (like PIL images), output in HSV
    Args: 
        image (PIL image)
    Returns:
        Image in CV2 image HSV
    """

    i = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    #return cv2.cvtColor(i,cv2.COLOR_BGR2YUV)
    return cv2.cvtColor(i,cv2.COLOR_BGR2HSV)

def CV2image2YUV(image):
    """
    Change the color space of the image to YUV.  Input image is in BGR space (like default cv2 images)
    Args: 
        image (BGR image)
    Returns:
        Image in YUV color space
    """
    return cv2.cvtColor(image,cv2.COLOR_BGR2YUV)


def switchRGB(image):
    """
    Switches image space from BGR to RGB space.  Default images when created in CV2 are in BGR space
    Args:
        image (BGR space)
    Returns:
        image (RGB space )
    """
    b,g,r = cv2.split(image)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])
    return rgb_img

