import numpy as np
import scipy.stats as stats


class NaiveBayes(object):
    def __init__(self):
        self._trained = False
        self._mu_vec_malignant_learned = []
        self._std_vec_malignant_learned = []
        self._mu_vec_benign_learned = []
        self._std_vec_benign_learned = []
        self._p_prior_benign_learned = None
        self._p_prior_malignant_learned = None

    def _mu_MLE(self, x):
        return (1/x.shape[0])*sum(x)

    def _sigma_square_MLE(self, x):
        res = 0
        n = x.shape[0]
        mu = self._mu_MLE(x)
        for k in range(x.shape[0]):
            res += (x[k] - mu)**2
        return (1/n) * res

    def _p_prior_malignant(self, y):
        return np.count_nonzero(y)/len(y)

    def _p_prior_benign(self, y):
        return 1 - self._p_prior_malignant(y)

    def train(self, x, y):
        malignant_index = np.where(y == 1)[0]
        benign_index = np.where(y == 0)[0]
        self._p_prior_malignant_learned = self._p_prior_malignant(y)
        self._p_prior_benign_learned = self._p_prior_benign(y)
        for feature in range(x.shape[0]):
            self._mu_vec_benign_learned.append(self._mu_MLE(x[feature, benign_index]))
            self._std_vec_benign_learned.append(np.sqrt(self._sigma_square_MLE(x[feature, benign_index])))
            self._mu_vec_malignant_learned.append(self._mu_MLE(x[feature,malignant_index]))
            self._std_vec_malignant_learned.append(np.sqrt(self._sigma_square_MLE(x[feature, malignant_index])))
        self._trained = True

    def classify_sample(self, x, sample):
        malignant = self._p_prior_malignant_learned
        benign = self._p_prior_benign_learned
        for feature in range(x.shape[0]):
            mu_benign = self._mu_vec_benign_learned[feature]
            std_benign = self._std_vec_benign_learned[feature]
            mu_malignant = self._mu_vec_malignant_learned[feature]
            std_malignant = self._std_vec_malignant_learned[feature]
            malignant = malignant * stats.norm.pdf(x[feature, sample], loc=mu_malignant, scale=std_malignant)
            benign = benign * stats.norm.pdf(x[feature, sample], loc=mu_benign, scale=std_benign)
        if malignant > benign:
            return 1
        else:
            return 0

    def classify(self, x):
        res = []
        for sample in range(x.shape[1]):
            res.append(self.classify_sample(x, sample))
        return res

    def evaluate(self, x, y):
        if not self._trained:
            print("Error : Please train before evaluating!")
        else:
            classified = self.classify(x)
            error = 0
            n = x.shape[1]
            for sample in range(n):
                if classified[sample] != y[sample]:
                    error += 1
            return error/n
