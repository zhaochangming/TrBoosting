import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy.special import softmax
from copy import deepcopy
from sklearn.metrics import log_loss, accuracy_score,balanced_accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from math import ceil, sqrt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import normalize
from sklearn.svm import SVC


MAX_RESPONSE = 4
MACHINE_EPSILON = np.finfo(np.float64).eps
PATENCE = 500
N_JOBS = 1


def mixup(a_x, a_y, b_x, b_y, alpha=0.75):
    x = np.zeros(a_x.shape)
    y = np.zeros(a_y.shape)
    mix_lambda = np.random.beta(alpha, alpha, len(a_x))
    for i in range(len(a_x)):
        mix_lambda[i] = max(mix_lambda[i], 1 - mix_lambda[i])
        x[i] = mix_lambda[i] * a_x[i] + (1 - mix_lambda[i]) * b_x[i]
        y[i] = mix_lambda[i] * a_y[i] + (1 - mix_lambda[i]) * b_y[i]
    return x, y


def sharpen(p, T=0.5):
    u = p ** (1 / T)
    return u / u.sum(dim=1, keepdim=True)


class weightRidge:
    def __init__(self, alpha):
        self.model = Ridge(alpha=alpha)
        self.alpha = alpha

    def fit(self, X, y, weight=None):
        self.model.fit(X, y, weight)

    def predict(self, X):
        if X.ndim == 2:
            return self.model.predict(X)
        else:
            return self.model.predict(X.reshape(1, -1))


class weightKernelRidge:
    def __init__(self, alpha):
        self.model = KernelRidge(alpha=alpha, kernel='rbf')
        self.alpha = alpha

    def fit(self, X, y, weight=None):
        self.model.fit(X, y, weight)

    def predict(self, X):
        if X.ndim == 2:
            return self.model.predict(X)
        else:
            return self.model.predict(X.reshape(1, -1))


class weightKNeighborsRegressor:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, algorithm='brute', weights='distance')
        self.n_neighbors = n_neighbors

    def fit(self, X, y, weight=None):
        resample_index = np.random.choice(range(len(X)), size=len(X), replace=True, p=weight / weight.sum())
        self.model.fit(X[resample_index], y[resample_index])

    def predict(self, X):
        if X.ndim == 2:
            return self.model.predict(X)
        else:
            return self.model.predict(X.reshape(1, -1))


class weightSVR:
    def __init__(self, C, gamma):
        self.model = SVR(kernel='rbf', C=C, gamma=gamma)
        self.C = C
        self.gamma = gamma

    def fit(self, X, y, weight=None):
        self.model.fit(X, y, weight)

    def predict(self, X):
        if X.ndim == 2:
            return self.model.predict(X)
        else:
            return self.model.predict(X.reshape(1, -1))


class node_generator:
    def __init__(self, actFunction='relu', featureDownSample='total'):
        self.Wlist = []
        self.blist = []
        self.scaler = []
        self.nonlinear = {
            'linear': self.linear,
            'sigmoid': self.sigmoid,
            'tanh': self.tanh,
            'relu': self.relu
        }[actFunction]
        self.max_iter = 0
        self.featureDownSample = featureDownSample

    @staticmethod
    def sigmoid(data):
        return 1.0 / (1 + np.exp(-data))

    @staticmethod
    def linear(data):
        return data

    @staticmethod
    def tanh(data):
        return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))

    @staticmethod
    def relu(data):
        return np.maximum(data, 0)

    @staticmethod
    def generator(shape):
        W = 2 * random.random(size=shape) - 1
        b = 2 * random.random() - 1
        return W, b

    def generator_nodes(self, data, batchsize):
        #
        fea_dim = data.shape[1]
        W_, b_ = self.generator((fea_dim, batchsize))
        if self.featureDownSample == 'total':
            mask_dim = fea_dim
        elif self.featureDownSample == 'sqrt':
            mask_dim = ceil(sqrt(fea_dim))
        elif self.featureDownSample == 'two':
            mask_dim = 2
        elif self.featureDownSample == 'tripleRoot':
            mask_dim = ceil(fea_dim ** 0.33)
        else:
            mask_dim = fea_dim
        mask_index = np.random.choice(range(fea_dim), mask_dim, replace=False)
        mask = np.zeros(fea_dim, dtype='bool')
        mask[mask_index] = True
        W_[~mask] = 0.
        #
        scaler_ = StandardScaler()  # QuantileTransformer(output_distribution='normal')
        data_ = np.dot(data, W_)
        scaler_.fit(data_ + b_)
        nodes = self.nonlinear(scaler_.transform(data_ + b_))
        self.Wlist.append(W_)
        self.blist.append(b_)
        self.scaler.append(scaler_)
        self.max_iter += 1
        return nodes

    def transform(self, testdata, iter=None):
        if iter is None:
            iter = self.max_iter
        self.testdata = testdata
        testNodes = Parallel(n_jobs=min(N_JOBS, iter))(delayed(self.transform_one_iter)(None, i) for i in range(iter))
        del self.testdata
        return testNodes

    def transform_one_iter(self, testdata=None, iter=None):
        if testdata is None:
            return self.nonlinear(self.scaler[iter].transform(self.testdata.dot(self.Wlist[iter]) + self.blist[iter]))
        else:
            return self.nonlinear(self.scaler[iter].transform(testdata.dot(self.Wlist[iter]) + self.blist[iter]))


class Early_stopping:
    def __init__(self, proof='loss', patience=10):
        self.proof = proof
        self.patience = patience
        self.count = 0
        self.lowestloss = None
        self.bestacc = None
        self.bestiter = None
        self.iter = 0

    def call_stopping(self, value):
        self.iter += 1
        if self.proof == 'loss':
            if self.lowestloss == None:
                self.lowestloss = value
                return False
            elif value < self.lowestloss:
                self.lowestloss = value
                self.count = 0
                self.bestiter = self.iter
                return False
            else:
                self.count += 1
                if self.count > self.patience:
                    return self.bestiter
        elif self.proof == 'acc':
            if self.bestacc == None:
                self.bestacc = value
                return False
            elif value > self.bestacc:
                self.bestacc = value
                self.count = 0
                self.bestiter = self.iter
                return False
            else:
                self.count += 1
                if self.count > self.patience:
                    return self.bestiter
        else:
            print(
                "I don't know the proof for early stopping, so there is no early_stopping. Please check."
            )


class BoostingBLS(node_generator):
    def __init__(self,
                 newNodeSize=10,
                 actFunction='relu',
                 boostingTimes=100,
                 verbose=False,
                 early_stopping=False, boostingModel='ridge', featureDownSample='total', batch_size=512, learning_rate=1.0):
        node_generator.__init__(self, actFunction=actFunction, featureDownSample=featureDownSample)
        self.newNodeSize = newNodeSize
        self.estimators = []
        self.iter = boostingTimes
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.trainloss = []
        self.valloss = []
        self.trainacc = []
        self.valacc = []
        self.boostingModel = boostingModel
        self.totalIter = 0
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    @staticmethod
    def _weight_and_response(y, prob, source_true_index=None):
        prob_ = np.copy(prob)
        prob_ = np.clip(prob_, a_min=0.0001, a_max=0.9999)
        sample_weight = prob_ * (1. - prob_)
        sample_weight = np.maximum(sample_weight, 2. * MACHINE_EPSILON)
        with np.errstate(divide="ignore", over="ignore"):
            z = np.where(y, 1. / prob_, -1. / (1. - prob_))
        z = np.clip(z, a_min=-MAX_RESPONSE, a_max=MAX_RESPONSE)
        if source_true_index is not None:
            sample_weight[~source_true_index] = 0.0
        return sample_weight, z

    def predict_score(self, node_model, X):
        if self.n_classes == 2:
            new_scores = node_model.predict(X)
            new_scores = np.c_[-new_scores, new_scores]
            new_scores = np.clip(new_scores, a_min=-MAX_RESPONSE, a_max=MAX_RESPONSE)
        else:
            new_scores = [e.predict(X) for e in node_model]
            new_scores = np.asarray(new_scores).T
            new_scores -= new_scores.mean(keepdims=True, axis=1)  # Friedman tell us
            new_scores *= (self.n_classes - 1) / self.n_classes
        new_scores = np.clip(new_scores, a_min=-MAX_RESPONSE, a_max=MAX_RESPONSE)
        # L2 norm
        new_scores = normalize(new_scores)
        return new_scores * self.learning_rate

    def get_balance_index(self, y, class_=None, weight=None):
        if weight is None:
            weight = np.ones(len(y))
        # balance index
        balance_index = []
        if class_ == -1:
            for c in range(self.n_classes):
                index_c = np.where(y[:, c] == 1)[0]
                weight_c = weight[index_c]
                balance_index.extend(self.down_sample(index_c, weight_c))
        else:
            balance_index_ = []
            for c in range(self.n_classes):
                index_c = np.where(y[:, c] == 1)[0]
                if len(index_c) == 0:
                    continue
                weight_c = weight[index_c]
                if c == class_:
                    balance_index.extend(self.down_sample(index_c, weight_c))
                else:
                    balance_index_.extend(self.down_sample(index_c, weight_c))
            balance_index.extend(random.choice(balance_index_, self.batch_size, replace=False))
        return balance_index

    def down_sample(self, index, weight, bs=None):
        if weight.sum() == 0:
            prob = None
        else:
            prob = weight / weight.sum()
        if bs is None:
            return random.choice(index, self.batch_size, replace=True, p=prob)
        else:
            return random.choice(index, bs, replace=True, p=prob)

    def predict(self, X, iter=None):
        if iter is None:
            iter = self.iter
        enchanted_list = self.transform(X, iter)
        output_ = Parallel(n_jobs=min(N_JOBS, iter))(
            delayed(self.predict_score)(self.estimators[i], enchanted_list[i]) for i in range(iter))
        output = np.zeros((len(X), self.n_classes))
        for i in range(iter):
            output += output_[i]
        del output_
        return output

    def boosting_fit(self, source_X=None, source_output=None, source_y=None, eval_data=None, eval_output=None, target_X=None,
                     target_y=None, target_output=None, unlabeled_target_X=None, unlabeled_target_output=None):
        eval_X, eval_y = None, None
        self.n_classes = target_output.shape[1]
        if source_X is not None:
            source_prob = np.where(self.n_classes == 2, np.array(softmax(source_output / 2, axis=1)),
                                   np.array(softmax(source_output, axis=1)))
        target_prob = np.where(self.n_classes == 2, np.array(softmax(target_output / 2, axis=1)),
                               np.array(softmax(target_output, axis=1)))
        unlabeled_y_augment = None
        unlabeled_prob_orign = None
        if self.boostingModel == 'ridge':
            model = weightRidge(alpha=0.001)  # 0.001
        elif self.boostingModel == 'svr':
            model = weightSVR(C=1, gamma='auto')
        elif self.boostingModel == 'knr':
            model = weightKNeighborsRegressor(n_neighbors=5)
        elif self.boostingModel.split('|')[0] == 'svr' and len(self.boostingModel.split('|')) == 3:
            C = int(self.boostingModel.split('|')[1])
            gamma = self.boostingModel.split('|')[2]
            if gamma == 'auto':
                model = weightSVR(C=C, gamma=gamma)
            else:
                gamma = float(gamma)
                model = weightSVR(C=C, gamma=gamma)
        else:
            model = weightRidge(alpha=0.001)  # 0.001
        if self.early_stopping:
            early_stoper = Early_stopping(proof='acc', patience=PATENCE)
        else:
            early_stoper = None
        if self.verbose:
            if source_X is not None:
                print('Init Source Loss: {}, ACC: {}'.format(
                    log_loss(source_y, source_prob),
                    accuracy_score(source_y.argmax(axis=1), source_prob.argmax(axis=1))))
            print('Init Target Loss: {}, ACC: {}'.format(
                log_loss(target_y, target_prob), accuracy_score(target_y.argmax(axis=1), target_prob.argmax(axis=1))))
        if eval_data is not None:
            (eval_X, eval_y) = eval_data
            self.eval_prob = np.where(self.n_classes == 2, np.array(softmax(eval_output / 2, axis=1)),
                                      np.array(softmax(eval_output, axis=1)))
            self.valacc.append(balanced_accuracy_score(eval_y.argmax(axis=1), self.eval_prob.argmax(axis=1)))
            if self.verbose:
                print('Init valLoss: {}, valACC: {}'.format(log_loss(eval_y, self.eval_prob), self.valacc[-1]))

        for i in range(self.iter):
            if source_X is not None:
                source_true_index = (self.classes_[np.argmax(source_y, axis=1)] == self.classes_[np.argmax(source_prob, axis=1)])
            self.totalIter += 1
            if source_X is not None:
                source_enhencedata = self.generator_nodes(source_X, self.newNodeSize)
            else:
                _ = self.generator_nodes(np.r_[target_X, unlabeled_target_X], self.newNodeSize)
            target_enhencedata = None
            unlabeled_enhencedata_origin = None
            unlabeled_enhencedata_augment = None
            unlabeled_prob_augment = None

            if target_X is not None:
                target_enhencedata = self.transform_one_iter(target_X, i)
            if unlabeled_target_X is not None:
                if self.noise_scale == 0:
                    noise = 0.0
                else:
                    noise = np.random.normal(0.0, self.noise_scale * self.feature_std, size=unlabeled_target_X.shape)

                unlabeled_target_X_augment = unlabeled_target_X + noise
                unlabeled_target_X_origin = unlabeled_target_X
                unlabeled_enhencedata_augment = self.transform_one_iter(unlabeled_target_X_augment, i)
                unlabeled_enhencedata_origin = self.transform_one_iter(unlabeled_target_X_origin, i)

                if i == 0:
                    unlabeled_initOutput_augment = self.get_initOutput(unlabeled_target_X_augment)
                    unlabeled_output_origin = self.get_initOutput(unlabeled_target_X_origin)
                    unlabeled_y_augment = self.labelEncoder.transform(
                        unlabeled_initOutput_augment.argmax(axis=1).reshape((-1, 1)))
                    unlabeled_y_origin = self.labelEncoder.transform(
                        unlabeled_output_origin.argmax(axis=1).reshape((-1, 1)))
                    unlabeled_prob_orign = np.where(self.n_classes == 2, np.array(softmax(unlabeled_output_origin / 2, axis=1)),
                                                    np.array(softmax(unlabeled_output_origin, axis=1)))
                    # print(log_loss(eval_y,unlabeled_prob_orign))
                    unlabeled_prob_augment = np.where(self.n_classes == 2,
                                                      np.array(softmax(unlabeled_initOutput_augment / 2, axis=1)),
                                                      np.array(softmax(unlabeled_initOutput_augment, axis=1)))
                else:
                    # noise label
                    unlabeled_output_augment = unlabeled_output_origin + unlabeled_new_scores_augment
                    # unlabeled_y_augment = self.labelEncoder.transform(unlabeled_output_augment.argmax(axis=1).reshape((-1, 1)))
                    unlabeled_prob_augment = np.where(self.n_classes == 2,
                                                      np.array(softmax(unlabeled_output_augment / 2, axis=1)),
                                                      np.array(softmax(unlabeled_output_augment, axis=1)))
                    # original prob
                    unlabeled_output_origin += unlabeled_new_scores_origin
                    unlabeled_prob_orign = np.where(self.n_classes == 2, np.array(softmax(unlabeled_output_origin / 2, axis=1)),
                                                    np.array(softmax(unlabeled_output_origin, axis=1)))
                    unlabeled_y_origin = self.labelEncoder.transform(unlabeled_output_origin.argmax(axis=1).reshape((-1, 1)))

            if self.n_classes == 2:
                weight, z = self._weight_and_response(source_y[:, 1], source_prob[:, 1], source_true_index=source_true_index)
                balance_index = self.get_balance_index(source_y, class_=-1, weight=weight)
                X_train, z_train, weight_train = source_enhencedata[balance_index], z[balance_index], weight[balance_index]
                new_estimators_ = deepcopy(model)  # must deepcopy the model!
                if target_X is not None:
                    target_weight, target_z = self._weight_and_response(target_y[:, 1], target_prob[:, 1])
                    target_balance_index = self.get_balance_index(target_y, class_=-1, weight=weight)
                    target_X_train, target_z_train, target_weight_train = target_enhencedata[target_balance_index], target_z[
                        target_balance_index], target_weight[target_balance_index]
                    X_train = np.r_[X_train, target_X_train]
                    weight_train = np.r_[weight_train, target_weight_train]
                    z_train = np.r_[z_train, target_z_train]
                new_estimators_.fit(X_train, z_train)
            else:
                new_estimators_ = []
                for j in range(self.n_classes):
                    model_copy = deepcopy(model)  # must deepcopy the model!
                    target_weight, target_z = self._weight_and_response(target_y[:, j], target_prob[:, j])
                    target_balance_index = self.get_balance_index(target_y, class_=j, weight=target_weight)
                    target_X_train, target_z_train, target_weight_train = target_enhencedata[target_balance_index], target_z[
                        target_balance_index], target_weight[target_balance_index]

                    if source_X is not None and (i % 2 == 0 or unlabeled_target_X is None):
                        source_weight, source_z = self._weight_and_response(source_y[:, j], source_prob[:, j],
                                                                            source_true_index=source_true_index)
                        source_balance_index = self.get_balance_index(source_y, class_=j, weight=source_weight)
                        source_X_train = source_enhencedata[source_balance_index]
                        source_z_train = source_z[source_balance_index]
                        X_train = np.r_[target_X_train, source_X_train]
                        z_train = np.r_[target_z_train, source_z_train]
                    else:
                        unlabeled_weight, unlabeled_z = self._weight_and_response(unlabeled_y_origin[:, j],
                                                                                  unlabeled_prob_augment[:, j])
                        unlabeled_balance_index = self.get_balance_index(unlabeled_y_origin, class_=j, weight=unlabeled_weight)
                        if len(unlabeled_balance_index) == 0:  # some class may not appear
                            X_train = target_X_train
                            z_train = target_z_train
                        else:
                            unlabeled_X_train = unlabeled_enhencedata_augment[unlabeled_balance_index]
                            unlabeled_z_train = unlabeled_z[unlabeled_balance_index]
                            X_train = np.r_[target_X_train, unlabeled_X_train]
                            z_train = np.r_[target_z_train, unlabeled_z_train]
                    model_copy.fit(X_train, z_train)
                    new_estimators_.append(model_copy)
            if unlabeled_target_X is not None:
                unlabeled_new_scores_origin = self.predict_score(new_estimators_, unlabeled_enhencedata_origin)
                unlabeled_new_scores_augment = self.predict_score(new_estimators_, unlabeled_enhencedata_augment)
            if source_X is not None:
                source_new_scores = self.predict_score(new_estimators_, source_enhencedata)
                source_output += source_new_scores
                source_prob = np.where(self.n_classes == 2, np.array(softmax(source_output / 2, axis=1)),
                                       np.array(softmax(source_output, axis=1)))
            target_new_scores = self.predict_score(new_estimators_, target_enhencedata)
            target_output += target_new_scores
            target_prob = np.where(self.n_classes == 2, np.array(softmax(target_output / 2, axis=1)),
                                   np.array(softmax(target_output, axis=1)))
            self.estimators.append(new_estimators_)

            #change
            # if (i+1)%10==0:
            #     source_y=np.eye(source_y.shape[1])[source_prob.argmax(axis=1)]
            #

            if self.verbose:
                if source_X is not None:
                    print('Source Iteration: {} Loss: {}, ACC: {}'.format(i + 1, log_loss(source_y, source_prob),
                                                                          accuracy_score(source_y.argmax(axis=1),
                                                                                         source_prob.argmax(axis=1))))
                print('Target Iteration: {} Loss: {}, ACC: {}'.format(i + 1,
                                                                      log_loss(target_y, target_prob),
                                                                      accuracy_score(target_y.argmax(axis=1),
                                                                                     target_prob.argmax(axis=1))))

            if eval_data is not None:
                eval_enhencedata = self.transform_one_iter(eval_X, i)
                eval_output += self.predict_score(new_estimators_, eval_enhencedata)
                self.eval_prob = np.where(self.n_classes == 2, np.array(softmax(eval_output / 2, axis=1)),
                                          np.array(softmax(eval_output, axis=1)))
                self.valacc.append(balanced_accuracy_score(eval_y.argmax(axis=1), self.eval_prob.argmax(axis=1)))
                if self.verbose:
                    print('Eval Iteration: {} valLoss: {}, valACC: {}'.format(i + 1, log_loss(eval_y, self.eval_prob),
                                                                              self.valacc[-1]))
                if self.early_stopping:
                    bestiter = early_stoper.call_stopping(self.valacc[-1])
                    if bestiter:
                        self.iter = bestiter
                        break


class traBroadNet(BoostingBLS, ClassifierMixin, BaseEstimator):
    def __init__(self,
                 boostingTimes=10,
                 actFunction='relu',
                 newNodeSize=10,
                 regL2=0.001,
                 normalize=True,
                 verbose=False,
                 early_stopping=False,
                 boostingModel='ridge', featureDownSample='total', batch_size=512, learning_rate=1.0, noise_scale=1.0):
        BoostingBLS.__init__(
            self,
            newNodeSize=newNodeSize,
            actFunction=actFunction,
            boostingTimes=boostingTimes,
            verbose=verbose,
            early_stopping=early_stopping,
            boostingModel=boostingModel, featureDownSample=featureDownSample, batch_size=batch_size, learning_rate=learning_rate)
        self.reg = regL2
        self.normalize = normalize
        self.noise_scale = noise_scale
        if self.normalize:
            self.normalScaler = StandardScaler()
            # self.normalScaler = QuantileTransformer(output_distribution='normal')
        self.labelEncoder = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.initLearner = None
        self.classes_ = None

    def get_initOutput(self, X):
        initOutput = self.initLearner.decision_function(X)
        if initOutput.ndim > 1:
            initOutput -= initOutput.mean(keepdims=True, axis=1)
            initOutput = normalize(initOutput)
        else:
            initOutput = normalize(np.c_[-initOutput, initOutput])[:, 1]
        return initOutput

    def decision_function(self, X, output=None, iter=None):
        if iter is None:
            iter = self.iter
        if self.normalize:
            X = self.normalScaler.transform(X)
        if output is None:
            output = self.get_initOutput(X)
        output_boosting = super(traBroadNet, self).predict(X, iter)
        if self.n_classes == 2:
            return np.c_[-output, output] + output_boosting
        else:
            return output + output_boosting

    def predict(self, X, output=None, iter=None):
        decision = self.decision_function(X, output, iter)
        return self.classes_[np.argmax(decision, axis=1)]

    def predict_prob(self, X, output=None, iter=None):
        if output==None:
            decision = self.decision_function(X, None, iter)
        else:
            decision = self.decision_function(X, np.copy(output), iter)
        return np.where(self.n_classes == 2, np.array(softmax(decision / 2, axis=1)), np.array(softmax(decision, axis=1)))

    def fit(self, source_X=None, source_y=None, source_initOutput=None, target_X=None, target_y=None, target_initOutput=None,
            eval_data=None, eval_output=None, initLearner=None, unlabeled_target_X=None, unlabeled_initOutput=None,
            unlabeled_initOutput_augment=None, unlabeled_initOutput_origin=None):
        self.classes_, _ = np.unique(source_y, return_inverse=True)                                                       # have changed !!!!!!!!!!!!!!
        if source_y is not None:
            source_y = self.labelEncoder.fit_transform(np.mat(source_y).T)
        else:
            self.labelEncoder.fit_transform(np.mat(target_y).T)
        if eval_data is not None:
            eval_data = (eval_data[0], self.labelEncoder.transform(eval_data[1].reshape((-1, 1))))
        if target_y is not None:
            target_y = self.labelEncoder.transform(target_y.reshape((-1, 1)))
        if self.normalize:
            if source_X is not None:
                source_X = self.normalScaler.fit_transform(source_X)
            if eval_data is not None:
                eval_data = (self.normalScaler.transform(eval_data[0]), eval_data[1])
            if target_X is not None:
                target_X = self.normalScaler.transform(target_X)
            if unlabeled_target_X is not None:
                unlabeled_target_X = self.normalScaler.transform(unlabeled_target_X)

        if (source_initOutput is not None):
            if source_y.shape[1] == 2:
                source_initOutput = np.c_[-source_initOutput, source_initOutput]
                if eval_data is not None:
                    eval_output = np.c_[-eval_output, eval_output]
                if target_initOutput is not None:
                    target_initOutput = np.c_[-target_initOutput, target_initOutput]
        else:
            if target_y.shape[1] == 2:
                if initLearner is None:
                    self.initLearner = SVC()
                    self.initLearner.fit(source_X, source_y[:, 1])
                else:
                    self.initLearner = initLearner
                source_initOutput = self.get_initOutput(source_X)
                source_initOutput = np.c_[-source_initOutput, source_initOutput]
                if eval_data is not None:
                    eval_output = self.get_initOutput(eval_data[0])
                    eval_output = np.c_[-eval_output, eval_output]
                if target_X is not None:
                    target_initOutput = self.get_initOutput(target_X)
                    target_initOutput = np.c_[-target_initOutput, target_initOutput]
                if unlabeled_target_X is not None:
                    unlabeled_initOutput = self.get_initOutput(unlabeled_target_X)
                    unlabeled_initOutput = np.c_[-unlabeled_initOutput, unlabeled_initOutput]
            else:  # 澶氬垎绫?
                if initLearner is None:
                    self.initLearner = SVC()
                    self.initLearner.fit(source_X, self.classes_[np.argmax(source_y, axis=1)])
                else:
                    self.initLearner = initLearner
                if source_X is not None:
                    source_initOutput = self.get_initOutput(source_X)
                if eval_data is not None:
                    eval_output = self.get_initOutput(eval_data[0])
                if target_X is not None:
                    target_initOutput = self.get_initOutput(target_X)
        if unlabeled_target_X is not None:
            self.feature_std = np.std(np.r_[target_X, unlabeled_target_X], axis=0)
        super(traBroadNet, self).boosting_fit(source_X=source_X, source_output=source_initOutput, source_y=source_y,
                                              eval_data=eval_data, eval_output=eval_output, target_X=target_X, target_y=target_y,
                                              target_output=target_initOutput, unlabeled_target_X=unlabeled_target_X,
                                              unlabeled_target_output=None)
