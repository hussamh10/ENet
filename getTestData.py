from cv2 import imread
import numpy as np

def getImage(i, source, main_dir):
    name = 'v' + str(i) + '.jpg'

    print(main_dir + source + '\\' + name)

    img = imread(main_dir + source + '\\' + name, 0)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.reshape(224, 224, 1)
    img = img.astype('float32')
    img /= 255
    return img

def getData(end, start = 1):
    input = []

    in1_src = ''
    in2_src = ''
    main = 'data\\4'

    for i in range(start, end):
        in1 = getImage(i, in1_src, main)
        in2 = getImage(i, in2_src, main)
        
        in_main = np.concatenate([in1, in2], axis = 2)
            
        input.append(in_main)
        
    input = np.array(input)

    return input, None
