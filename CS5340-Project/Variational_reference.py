import numpy as np
import cv2
from functions import io_data
import copy
import matplotlib.pyplot as plt


class IsingModel():
    def __init__(self, height, width, w, c, img_noise):
        """

        :param height: height of input image
        :param width: width of input image
        :param w: coupling strenght in Ising model
        :param c:  variance of guassian model for uniary term, namely p(y/x)
        :param img_noise: corrupted image
        """

        self.width, self.height = height, width
        self.mu = img_noise
        self.grid = img_noise
        self.observed = img_noise
        self.w = w
        self.c = c

    def neighbours(self, x, y):
        n = []
        if x == 0:
            n.append((self.width - 1, y))
        else:
            n.append((x - 1, y))
        if x == self.width - 1:
            n.append((0, y))
        else:
            n.append((x + 1, y))
        if y == 0:
            n.append((x, self.height - 1))
        else:
            n.append((x, y - 1))
        if y == self.height - 1:
            n.append((x, 0))
        else:
            n.append((x, y + 1))
        return n

    def local_sum(self, x, y):

        neighbors = self.neighbours(x, y)
        summ = sum(self.w * self.mu[xx, yy] for (xx, yy) in neighbors)
        return  summ

    def gibbs_move(self):


        for x in range(self.width):
            for y in range(self.height):
                m_i = self.local_sum(x,y)
                l_1 = gaussian(1.0, self.grid[x,y], self.c)
                l_0 = gaussian(-1.0, self.grid[x, y], self.c)

                a_i = m_i + 0.5 * (l_1 - l_0)
                q_1 = sigm(2 * a_i)
                q_0 = sigm(-2 * a_i)
                self.mu[x, y] = tanh(a_i)

                if q_1 > 0.5:
                    self.grid[x, y] = 1.0
                else:
                    self.grid[x, y] = -1.0


def tanh(x):

    return (np.exp(-x)/(np.exp(x)+np.exp(-x)))

def sigm(x):

    return (1.0/(1.0+np.exp(-x)))

def gaussian(x, mu, sigma):

    res = 1.0/(np.sqrt(2.0 * np.pi)* sigma)*np.exp(-(x - mu) **2 / (2.0 * sigma **2))

    return  res

def convert_img(img):

    img[img == 255] = 1.0
    img[img == 0] = -1.0

    return img

def convert_back(data):

    data[data == 1.0] = 255
    data[data == -1.0] = 0

    return  data

def main():
    J = 2
    sigma = 1
    for i in range(4):
        print('Working on image {}'.format(i+1))
        data, img_noise = io_data.read_data("a1/{}_noise.txt".format(i+1), True)
        img = np.squeeze(copy.deepcopy(img_noise))
        img = convert_img(img)
        Ismodel = IsingModel(img.shape[0], img.shape[1], J, sigma, img)
        Ismodel.grid = img
        for _ in range(5):
            Ismodel.gibbs_move()

        avg = Ismodel.grid

        cv2.imshow('{}_noise'.format(i+1), img_noise)
        img_denoise = np.expand_dims(convert_back(avg), axis = 2)
        cv2.imshow('{}_denoise'.format(i+1), img_denoise)
        cv2.waitKey()
        cv2.imwrite('./output/{}_denoise_vari.jpg'.format(i+1), img_denoise)


if __name__ == '__main__':

    main()






