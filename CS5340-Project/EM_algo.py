import numpy as np
import cv2
from functions import io_data
import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy.linalg import inv, det
import h5py
from scipy.stats import multivariate_normal
import random

class MixtureModel():
    def __init__(self, k, alpha, mu, sigma, n, data):
        """
        k: the number of gaussian kernels
        alpha : the mixture coefficients
        mu: mean of all gaussian kernels of shape[k, num_feature]
        sigma: variance of gaussian kernels of shape[k, num_feature, num_feature]
        n : number of data
        data: input data

        """
        self.k = k
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.res = np.zeros((n, k))
        self.data = data


    def E_step(self):

        """
        E step of the EM algo, updating the responsibilities with current parameters
        """

        res = []
        for j in range(self.k):
            try:
                res.append(np.expand_dims(self.alpha[j]*multivariate_normal.pdf(self.data, self.mu[j], self.sigma[j]), axis = 1))
            except ValueError:
                print (self.sigma)

        res = np.concatenate(res, axis = 1)
        self.res = res/np.sum(res, axis = 1, keepdims = True)


    def M_step(self):

        """
        M step of EM algo, update parameters with the responsibilities computed from E step
        """


        for j in range(self.k):
            N_k  =  np.sum(self.res, axis = 0)[j]
            res = np.expand_dims(self.res[:, j], axis = 1)
            self.mu[j, :] = np.sum(res * self.data, axis = 0)/N_k
            sigma = np.zeros_like(self.sigma[0])
            for i in range(self.data.shape[0]):
                sigma += self.res[i, j] * np.outer((self.data[i, :] - self.mu[j, :]), (self.data[i, :] - self.mu[j, :]))
            self.sigma[j, :, :] = sigma / N_k
            self.alpha[j] = N_k/self.data.shape[0]

    def prob(self):

        """
        compute the prob of the data using current parameters to decide convergence
        """
        res = []
        for j in range(self.k):
            res.append(np.expand_dims(self.alpha[j] * multivariate_normal.pdf(self.data, self.mu[j], self.sigma[j]),axis = 1))

        res = np.log(np.sum(np.hstack(res), axis = 1))
        log_pro = np.sum(res)
        return log_pro


def kmeans(k,  data, iter):

    """
    using kmeans to initialize the EM algo
    :param k: number of clusters in kmeans, the same value with the number of gaussian kernels in EM
    :param data:  input data of shape[n, c] n is the number of data, c is the number of feature
    :param iter:  number of iterations
    :return:  cluster center and label
    """
    center_index = random.sample(range(data.shape[0]), k)
    center = data[center_index]
    for n in range(iter):
        dist = (np.expand_dims(data, axis = 1) - np.expand_dims(center,0)) **2
        dist = np.sqrt(np.sum(dist, axis = 2))
        label = np.argmin(dist, axis = 1)
        for i in range(k):
            center[i, :] = np.mean(data[np.where(label == i)], axis = 0)


    return center, label


def convert_label_to_pi(label):
    """

    :param label: the label given by kmeans, use it to initialize the mixture coefficient of EM
    :return: alpha
    """


    label = np.array(label)
    alpha_1 = np.float32(np.count_nonzero(label))/label.shape[0]
    alpha_0 = 1 - alpha_1

    return np.array([alpha_0, alpha_1])

# def generate_res_img(res, img, name):
#
#     """
#
#     :param res: responsibilites returned from EM algo
#     :param img:
#     :param name:
#     :return:
#     """
#
#     res = res /np.sum(res, axis = 1, keepdims = True)
#     res0 = res[:, 1].reshape([img.shape[0], img.shape[1]])
#     res0[res0 >= 0.5] =1.0
#     res0[res0 < 0.5] = 0.0
#     img_fore  =  img * np.expand_dims(res0, axis = 2)
#     img_back = img * np.expand_dims(1.0-res0, axis=2)
#     img_mask = 1-res0
#
#     cv2.imshow(name, img)
#     cv2.imshow('{}_fore'.format(name),  img_fore)
#     cv2.imshow('{}_back'.format(name), img_back)
#     cv2.imshow('{}_mask'.format(name), img_mask)
#     cv2.waitKey()
#     cv2.imwrite('./output/{}_fore.jpg'.format(name) ,(img_fore * 255.0).astype(np.uint8))
#     cv2.imwrite('./output/{}_back.jpg'.format(name), (img_back * 255.0).astype(np.uint8))
#     cv2.imwrite('./output/{}_mask.jpg'.format(name), (img_mask*255).astype(np.uint8))


def generate_res_labdata(res, data, name):

    """

    :param res: responsibilities returned from EM algo
    :param data: original input data
    :param name: img name to save result
    :return: show and save all results
    """

    res = res /np.sum(res, axis = 1, keepdims = True)
    res0 = res[:, 1]
    res0[res0 >= 0.5] =1.0
    res0[res0 < 0.5] = 0.0
    data_img = copy.deepcopy(data[:, 2:])
    img_fore  =  data_img * np.expand_dims(res0, axis = 1)
    img_fore = np.concatenate([data[:, :2], img_fore], axis = 1)
    img_back = data_img * np.expand_dims(1.0-res0, axis=1)
    img_back = np.concatenate([data[:, :2], img_back], axis = 1)
    img_mask = np.expand_dims(1.0-res0, axis=1)
    img_mask = np.concatenate([data[:, :2], img_mask], axis =1 )
    io_data.write_data(img_fore, '../{}_fore.txt'.format(name))
    _, img = io_data.read_data('../{}_fore.txt'.format(name), is_RGB = False, visualize=True)
    io_data.write_data(img_back, '../{}_back.txt'.format(name))
    _, img = io_data.read_data('../{}_back.txt'.format(name), is_RGB =False, visualize=True)


def generate_mask(data, name):

    n = data.shape[0]
    data_img = copy.deepcopy(data[:, 2:])
    k = 2
    iter = 300
    center, label= kmeans(k, data_img, iter)
    label_0 = np.where(label == 0)
    label_1 = np.where(label == 1)

    sigma_0 = np.expand_dims(np.cov(data_img[label_0], rowvar = 0), axis = 0)
    sigma_1 = np.expand_dims(np.cov(data_img[label_1], rowvar = 0), axis = 0)
    sigma = np.concatenate([sigma_0, sigma_1], axis=0)


    alpha = convert_label_to_pi(label)
    Model = MixtureModel(k, alpha, center, sigma, n, data_img)
    c = 0
    prob_all = []
    thre = 0.00005
    while True:

        Model.E_step()
        Model.M_step()
        c += 1
        prob_all.append(Model.prob())
        print('Iteration : {}, Prop: {}'.format(c, Model.prob()))
        if len(prob_all) > 5:
            prob_last = np.array(prob_all)
            diff = prob_last[-5:] - prob_last[-6:-1]

            if np.all(diff < thre):
                generate_res_labdata(Model.res, data, name)
                break


if __name__ == '__main__':

    filename = ['cow', 'fox', 'owl', 'zebra' ]
    for name in filename:
        data, img = io_data.read_data("a2/{}.txt".format(name), True)
        generate_mask(data, name)






