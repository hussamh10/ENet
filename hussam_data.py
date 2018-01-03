from cv2 import imread
import numpy as np

import numpy as np
import cv2

size = 20

def getImage(i, source):
    name = str(i) + '.jpg'
    print('data\\' + source + '\\' + name)
    img = imread('data\\' + source + '\\' + name, 0)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype('float32')
    img /= 255
    return img

def getData(end, start=0):
    img_src = 'images'

    imgs = []
    labels = []

    for i in range(start, end):
        image1 = getImage(i, img_src)
        image2 = getImage(i, img_src)
        image1 = image1.reshape((224, 224, 1))
        image2 = image2.reshape((224, 224, 1))
        image = np.concatenate([image1, image2], axis=2)

        imgs.append(image)
        labels.append(getImage(i, img_src))
        
    imgs = np.array(imgs)
    labels = np.array(labels)

    return imgs, labels
