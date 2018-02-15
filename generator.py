import numpy as np
from cv2 import imread


def getImage(i, source, main_dir):
    name = str(i) + '.jpg'

    print(main_dir + source + '\\' + name)

    img = imread(main_dir + source + '\\' + name, 0)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.reshape(224, 224, 1)
    img = img.astype('float32')
    img /= 255
    return img

def generate(size):
    dir = 'data\\'
    src_x1 = '1\\'
    src_x2 = '2\\'
    src_x3 = '3\\'

    i = 0

    while i < size:
        x1 = getImage(i, src_x1, dir)
        x2 = getImage(i, src_x2, dir)
        x3 = getImage(i, src_x3, dir)

        x1 = x1.reshape((1, x1.shape[0], x1.shape[1], x1.shape[2]))
        x2 = x2.reshape((1, x2.shape[0], x2.shape[1], x2.shape[2]))
        x3 = x3.reshape((1, x3.shape[0], x3.shape[1], x3.shape[2]))

        x1_2 = np.concatenate([x1, x2], axis=3)

        i += 1

        yield(x1_2, x3)


