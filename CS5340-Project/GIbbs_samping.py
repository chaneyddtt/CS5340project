import numpy as np
import cv2
from functions import io_data
import copy
import matplotlib.pyplot as plt


class IsingModel():

##### Initialize Ising model , grid is updated every time, J is the coupling strength, sigma for gaussian #######
    def __init__(self, height, width, J, sigma, img_noise):
        self.width, self.height = height, width
        self.grid = img_noise
        self.observe = img_noise
        self.J = J
        self.sigma = sigma

##### TO find neighboring nodes in Ising Model #######
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

##### TO compute the sum of neighboring nodes in Ising Model #######
    def local_sum(self, x, y):

        neighbors = self.neighbours(x, y)
        summ = sum(self.grid[xx, yy] for (xx, yy) in neighbors)
        return  summ

##### TO compute the probabilility of a node #######
    def local_prob(self, x, y):

        evi = self.grid[x,y]
        prop_1 = np.exp(self.J * self.local_sum(x,y)) * gaussian(1.0, evi, self.sigma)
        prop_0 = np.exp(- self.J * self.local_sum(x,y)) * gaussian(-1.0, evi, self.sigma)

        return  prop_1 / (prop_0 + prop_1)


    def gibbs_move(self):

        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                p = self.local_prob( x, y)
                if np.random.random() <= p:
                    self.grid[x, y] = 1.0
                else:
                    self.grid[x, y] = -1.0



def gaussian(x, mu, sigma):

    res = 1/(np.sqrt(2.0 * np.pi)* sigma)*np.exp(-(x - mu) **2 / (2.0 * sigma **2))

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
    sigma = 2
    for i in range(4):
        print('Working on image {}'.format(i+1))
        data, img_noise = io_data.read_data("a1/{}_noise.txt".format(i+1), True)
        img = np.squeeze(copy.deepcopy(img_noise))
        img = convert_img(img)
        Ismodel = IsingModel(img.shape[0], img.shape[1], J, sigma, img)
        Ismodel.grid = img
        for _ in range(5):
            Ismodel.gibbs_move()
        avg= Ismodel.grid

        cv2.imshow('{}_noise'.format(i+1), img_noise)
        img_denoise = np.expand_dims(convert_back(avg), axis = 2)
        cv2.imshow('{}_denoise'.format(i+1), img_denoise)
        cv2.waitKey()
        cv2.imwrite('./output/{}_denoise_gibbs.jpg'.format(i+1), img_denoise)


if __name__ == '__main__':

   main()





