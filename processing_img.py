import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications.vgg19 import preprocess_input


###############################################################################
def clip_array(arr, a, b):
	arr = np.clip(arr, a, b).astype('uint8')
	return arr

###############################################################################
def preprocess_image(img):
    #img = load_img(img_path)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis = 0)
    return img

###############################################################################
def inverse_preprocess(x):
    # perform the opposite of the preprocessing step (preprocess_input function)
    # reason for the following values--> 
    # line 200 of (https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/applications/imagenet_utils.py#L97-L110)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = clip_array(x, 0, 255)
    return x

################################################################################
def show_image(image):
    if len(image.shape) == 4:
        image = np.squeeze(image, axis = 0)

    image = inverse_preprocess(image)
    #plt.imshow(image)
    #plt.show()
    return image

################################################################################
