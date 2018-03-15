from unet import get_unet
from hussam_data import getData as gd
from matplotlib import pyplot as plt
import keras


def test(start):
    md = get_unet()
    md.load_weights('enet_14_3.hdf5')

    h, hy = gd(start = start, end = start + 10, folder='_ (11)')

    print("predicting")

    hp = md.predict(h)

    i = start
    for p, gt in zip(hp, hy):
        plt.imshow(p.reshape((224, 224)), cmap='gray')
        plt.savefig('' + 'out\\' + str(i) + '.png')

        plt.imshow(gt.reshape((224, 224)), cmap='gray')
        plt.savefig('' + 'out\\gt' + str(i) + '.png')

        print(''  + 'out\\' + str(i) + '.png')
        i += 1

    print("Done")

test(1)
