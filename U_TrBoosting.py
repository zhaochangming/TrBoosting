from SS_TrBoosting import traBroadNet
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from math import ceil


class NNLinear:
    def __init__(self, W, bias):
        self.W = W
        self.bias = bias

    def decision_function(self, X):
        return X @ self.W + self.bias.reshape((1, -1))


def data_augmentation(X_target, y_target, initLearner, num_enhanced=100, n_class=None):
    if n_class is None:
        n_class = len(np.unique(y_target))
    y_enhanced = np.zeros((n_class * num_enhanced, n_class))
    label_enhanced = np.zeros(n_class * num_enhanced)
    i = 0
    for c in range(n_class):
        lambda_list = np.random.beta(0.75, 0.75, num_enhanced)
        x = np.arange(n_class)
        x = np.delete(x, c)
        for lambda_ in lambda_list:
            max_lambda = max(lambda_, 1 - lambda_)
            min_lambda = 1 - max_lambda
            y_enhanced[i, c] = max_lambda
            y_enhanced[i, np.random.choice(x, 1)] = min_lambda
            label_enhanced[i] = c
            i += 1

    reg = Ridge(alpha=0.1, fit_intercept=False)
    reg.fit(initLearner.W.T, (y_enhanced - initLearner.bias.reshape((1, -1))).T)
    X_enhanced = reg.coef_
    X_enhanced = (X_enhanced - X_enhanced.mean(axis=0)) / (X_enhanced.std(axis=0) + 1e-6)
    X_enhanced = X_enhanced * X_target.std(axis=0) + X_target.mean(axis=0)
    # print('enhance BCA: ',
    #       balanced_accuracy_score(label_enhanced, initLearner.decision_function(X_enhanced).argmax(axis=1)))
    return X_enhanced, label_enhanced


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)

    return entropy


def Usupervised_TrBoosting(X_target, y_target, initLearner, PS=0.8, NodeSize=20, batch_size=64):
    output = initLearner.decision_function(X_target)
    mean_ent = Entropy(nn.Softmax(dim=1)(torch.from_numpy(output)))
    cls_k = output.shape[1]
    predict = output.argmax(axis=1)
    value = mean_ent
    new_tar = []
    new_src = []
    for c in range(cls_k):
        c_idx = np.where(predict == c)
        c_idx = c_idx[0]
        c_value = value[c_idx]

        _, idx_ = torch.sort(c_value)
        c_num = len(idx_)
        c_num_s = ceil(c_num * PS)
        for ei in range(0, c_num_s):
            new_src.append(c_idx[idx_[ei]])
        for ei in range(c_num_s, c_num):
            new_tar.append(c_idx[idx_[ei]])
    few_short_index = new_src
    no_label_index = new_tar
    X_enhanced, label_enhanced = data_augmentation(X_target, predict[few_short_index], initLearner, n_class=4)
    estimator = traBroadNet(boostingTimes=100, actFunction='relu', newNodeSize=NodeSize, normalize=False, verbose=False,
                                batch_size=batch_size, learning_rate=0.01, noise_scale=0.1)
    estimator.fit(
        source_X=X_enhanced, source_y=label_enhanced,
        target_X=X_target[few_short_index], target_y=predict[few_short_index],
        eval_data=(X_target, y_target),
        initLearner=initLearner, unlabeled_target_X=X_target[no_label_index]
    )
    return estimator.valacc[-1], estimator.eval_prob, estimator












