
# import numpy as np
# from sklearn.model_selection import train_test_split
# from PIL import Image
from adaboost import AdaBoost


def adaboost_train(X_tr, Y_tr, T=5):
    model = AdaBoost(X_tr.T, Y_tr, M=T)
    model.train()
    return model


def adaboost_predict(model, X_te):
    pred = model.pred(X_te.T)
    return pred


# def read_data():
#     dataset = np.loadtxt('data/data.txt', delimiter=",")
#     x = dataset[:, 0:8]
#     y = dataset[:, 8]
#
#     # prepare train data
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#
#     # prepare test and train data
#     # x_train = x_train.transpose()
#     y_train[y_train == 1] = 1
#     y_train[y_train == 0] = -1
#
#     # x_test = x_test.transpose()
#     y_test[y_test == 1] = 1
#     y_test[y_test == 0] = -1
#     return x_train, y_train, x_test, y_test


# def accuracy_score(y_true, y_pred):
#     assert len(y_true) == len(y_pred)
#     score = (y_pred == y_true).astype(float).sum() / len(y_true)
#     return score


# def main():
#     X_tr, Y_tr, X_te, Y_te = read_data()
#     model = adaboost_train(X_tr, Y_tr, T=10)
#     pred = adaboost_predict(model, X_te)
#
#     print("total test: ", len(pred))
#     print("true pred: ", len(pred[pred == Y_te]))
#     print("acc: ", accuracy_score(Y_te, pred))
#
#     return


# if __name__ == '__main__':
#     main()
#     print('done!')

