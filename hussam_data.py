from cv2 import imread
import numpy as np

import numpy as np
import cv2

size = 20

def getImage(i, source):
    #i = (i%10) + 1
    name = str(i) + '.jpg'
    print('data\\' + source + '\\' + name)
    img = imread('data\\' + source + '\\' + name, 0)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype('float32')
    img /= 255
    return img

def getData(end, start=0):
    img_src = 'img'
    label_src = 'label'

    imgs = []
    labels = []

    for i in range(start, end):
        i+=1 #off by one
        imgs.append(getImage(i, img_src))
        labels.append(getImage(i, label_src))
        
    imgs = np.array(imgs)
    labels = np.array(labels)

    return imgs, labels
