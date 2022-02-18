import numpy as np
from classifier import DecisionStump
from collections import OrderedDict
from utils import accuracy_score


class AdaBoost:

    def __init__(self, X, y, M=5, weaker=DecisionStump):
        self.X = X
        self.y = y
        self.weaker = weaker
        self.sums = np.zeros(self.y.shape)

        self.W = np.ones((self.X.shape[1], 1)).flatten() / self.X.shape[1]
        self.M = M
        # self.Q = 0
        self.G = OrderedDict()
        self.alpha = OrderedDict()

    def train(self):
        for i in range(self.M):
            self.G[i] = self.weaker(self.X, self.y)
            e = self.G[i].train(self.W)
            self.alpha[i] = 1.0 / 2 * np.log((1 - e) / e)
            res = self.G[i].pred(self.X)

            # print("acc of weak classfier {}: {}".format(i+1, accuracy_score(self.y, res)))

            Z = self.W * np.exp(-self.alpha[i] * self.y * res.transpose())
            self.W = (Z / Z.sum()).flatten()
            # self.Q = i
            if (self.errorcnt(i) == 0):
                print("{}-th weak classifier can reduce the error rate to 0.".format(i + 1))
                break

    def errorcnt(self, t):
        self.sums = self.sums + self.G[t].pred(self.X).flatten() * self.alpha[t]

        pre_y = np.zeros(np.array(self.sums).shape)
        pre_y[self.sums >= 0] = 1
        pre_y[self.sums < 0] = -1

        t = (pre_y != self.y).sum()
        return t

    def pred(self, test_X):
        test_X = np.array(test_X)
        sums = np.zeros(test_X.shape[1])
        for i in range(len(self.G)):
            sums = sums + self.G[i].pred(test_X).flatten() * self.alpha[i]
        pre_y = np.zeros(np.array(sums).shape)
        pre_y[sums >= 0] = 1
        pre_y[sums < 0] = -1
        return pre_y

