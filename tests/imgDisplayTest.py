import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


def main():
    fig = plt.figure(figsize=(10, 10)) # size of images in INCHES           TODO: change to pixel-ratio

    origImg = mpimg.imread('pictures/Moi.jpg')

    imgResized1 = Image.open('pictures/Moi.jpg')
    imgResized1.thumbnail((64, 64), Image.ANTIALIAS)  # resizes image in-place

    imgResized2 = Image.open('pictures/Moi.jpg')
    imgResized2.thumbnail((64, 64), Image.ANTIALIAS)  # resizes image in-place

    imgOutput = mpimg.imread('pictures/Moi.jpg')

    imgs = [origImg, imgResized1, imgResized2, imgOutput]

    for i in range(len(imgs)):
        fig.add_subplot(2, 2, i+1)
        plt.imshow(imgs[i])

    plt.show()


if __name__ == '__main__':
    main()
