
import numpy as np


class DecisionStump:
    def __init__(self, X, y):
        # self.X = np.array(X)
        # self.y = np.array(y)
        self.X = X
        self.y = y
        self.N = self.X.shape[0]

    def train(self, W, steps=100):
        min = float('inf')
        threshold_value = 0
        threshold_pos = 0
        threshold_tag = 0
        self.W = np.array(W)
        for i in range(self.N):
            value, errcnt = self.findmin(i, 1, steps)
            if (errcnt < min):
                min = errcnt
                threshold_value = value
                threshold_pos = i
                threshold_tag = 1
        for i in range(self.N):  # -1
            value, errcnt = self.findmin(i, -1, steps)
            if (errcnt < min):
                min = errcnt
                threshold_value = value
                threshold_pos = i
                threshold_tag = -1
        self.threshold_value = threshold_value
        self.threshold_pos = threshold_pos
        self.threshold_res = threshold_tag
        # print('threshold_value: {}, threshold_pos: {}, threshold_res: {}'.format(self.threshold_value, self.threshold_pos, self.threshold_res))
        return min

    def findmin(self, i, tag, steps):
        t = 0
        tmp = self.predintrain(self.X, i, t, tag).transpose()
        errcnt = np.sum((tmp != self.y) * self.W)
        buttom = np.min(self.X[i, :])
        up = np.max(self.X[i, :])
        minerr = float('inf')
        value = 0
        st = (up - buttom) / steps
        for t in np.arange(buttom, up, st):
            tmp = self.predintrain(self.X, i, t, tag).transpose()
            errcnt = np.sum((tmp != self.y) * self.W)
            if errcnt < minerr:
                minerr = errcnt
                value = t
        return value, minerr

    def predintrain(self, test_set, i, t, tag):
        test_set = np.array(test_set).reshape(self.N, -1)
        pre_y = np.ones((np.array(test_set).shape[1], 1))
        pre_y[test_set[i, :] * tag < t * tag] = -1
        return pre_y

    def pred(self, test_X):
        test_X = np.array(test_X).reshape(self.N, -1)
        pre_y = np.ones((np.array(test_X).shape[1], 1))
        pre_y[test_X[self.threshold_pos, :] * self.threshold_res < self.threshold_value * self.threshold_res] = -1
        return pre_y
