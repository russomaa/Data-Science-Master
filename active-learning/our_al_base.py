# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:26:01 2017
Revised on Thu Feb 15 17:24:27 2018
@author: Jordi
A class defining the 'active object' and useful methods for it.
"""

from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import SVC
if sklearn.__version__ == '0.17':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split


def show_results(q, random, active, nfig=1):
    plt.figure(nfig)
    plt.clf()
    plt.subplot(2, 1, 1)
    random.makeplots(q)
    plt.title('Random')
    # plt.figure(2)
    plt.subplot(2, 1, 2)
    active.makeplots(q)
    plt.title('Active')
    plt.draw()
    plt.show(0)


class ao:
    """ AO (active object) class. """

    def __init__(self):
        self.colors = ['r', 'g', 'b', 'k', 'm']
        self.classifier = None
        self.xlab, self.ylab, self.xunlab, self.yunlab = None, None, None, None
        self.idx, self.acc = None, None
        self.gamma, self.C = 1.0, 100

    def setup(self, labeled_data, labels, unlabeled_data, labels_u, test_data, test_labels):
        """
        Setup training, test, labeled and unlabeled datasets.
        Returns the test dataset, for final validation.
        """
        # Split training data in labeled/unlabeled
        self.xlab = labeled_data
        self.ylab = labels
        self.xunlab = unlabeled_data
        self.yunlab = labels_u

        xtest = test_data
        ytest = test_labels

        # Classifier: SVM
        sigma = np.mean(pdist(self.xlab))
        self.gamma = 1 / (2 * sigma * sigma)
        self.classifier = SVC(C=self.C, gamma=self.gamma, decision_function_shape='ovr')
        self.idx = []
        self.acc = []

        return xtest, ytest

    def copy(self):
        """
        Create a copy of itself by creating a new ao object and copying contents.
        """
        copy = ao()
        copy.xlab = self.xlab.copy()
        copy.xunlab = self.xunlab.copy()
        copy.ylab = self.ylab.copy()
        copy.yunlab = self.yunlab.copy()
        copy.gamma = self.gamma
        copy.C = self.C
        copy.classifier = SVC(C=self.C, gamma=self.gamma, decision_function_shape='ovr')
        copy.idx = self.idx.copy()
        copy.acc = self.idx.copy()
        return copy

    def updateLabels(self, idx):
        """
        Move selected samples from unlabeled to labeled set.
        """
        self.xlab = np.concatenate((self.xlab, self.xunlab[idx, :]), axis=0)
        self.ylab = np.concatenate((self.ylab, self.yunlab[idx]), axis=0)
        self.xunlab = np.delete(self.xunlab, idx, axis=0)
        self.yunlab = np.delete(self.yunlab, idx, axis=0)
        # Save them
        self.idx.append(idx)

    def score(self, xtest, ytest):
        """ Compute score on xtest/ytest, appends to self.acc and returns estimated value. """
        acc = self.classifier.score(xtest, ytest)
        self.acc.append(acc)
        return acc

    # Here are convenient functions to show results graphically
    def scatter(self, plot_unlab=False, marker='o', ms=60, num_points=0, mec=None):
        """
        A scatter plot of unlabeled and labeled points.
        """
        if plot_unlab:
            plt.scatter(self.xunlab[:, 0], self.xunlab[:, 1], s=5, c='lightgray')
        xlab = self.xlab[-num_points:, :]
        ylab = self.ylab[-num_points:]
        plt.scatter(xlab[:, 0], xlab[:, 1], c=ylab,
                    marker=marker, s=ms, edgecolors=mec)
        plt.grid(1)

    def plot(self, plot_unlab=False, marker='o', ms=6, num_points=0, mec=None):
        """
        Plot unlabeled and labeled points.
        """
        if plot_unlab:
            plt.plot(self.xunlab[:, 0], self.xunlab[:, 1], '.', ms=2, color='gray')
        xlab = self.xlab[-num_points:, :]
        ylab = self.ylab[-num_points:]
        # for i,c in enumerate(np.unique(ylab)):
        for c in np.unique(ylab):
            plt.plot(xlab[ylab == c, 0], xlab[ylab == c, 1], c=self.colors[int(c)],
                     ls='None', ms=ms, marker=marker, mec=mec, mew=2)
        plt.grid(1)

    def plotdf(self):
        """
        Plot decision function for SVM. Only works for binary problems.
        """
        xtrain = np.concatenate((self.xlab, self.xunlab), axis=0)

        x_min = xtrain[:, 0].min()
        x_max = xtrain[:, 0].max()
        y_min = xtrain[:, 1].min()
        y_max = xtrain[:, 1].max()

        grid_size_x = (x_max - x_min)
        grid_size_y = (y_max - y_min)

        x_min -= grid_size_x * 0.2
        x_max += grid_size_x * 0.2
        y_min -= grid_size_y * 0.2
        y_max += grid_size_y * 0.2

        xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_size_x * .01),
                             np.arange(y_min, y_max, grid_size_y * .01))

        # z = self.classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
        z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)

        #plt.contourf(xx, yy, (z < 0) * 1, alpha = 0.4, levels = [0, 0.5, 1])
        plt.contourf(xx, yy, z, alpha=0.4) #, cmap=plt.cm.Accent)  # terrain is nice too
        # More colormaps at http://matplotlib.org/examples/color/colormaps_reference.html

    def makeplots(self, query_points):
        """
        Make plots showing selected samples and decision boundaries.
        """
        self.plotdf()
        self.plot(True)
        self.plot(marker='s', num_points=query_points, mec='k')
        # plt.plot(self.xlab[-1,0], self.xlab[-1,1], 'bs')
