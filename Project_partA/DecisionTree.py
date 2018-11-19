import numpy as np


class DecisionTree(object):

    def __init__(self, error_criterion: str, min_information_gain: int, max_depth: int):
        self._error_criterion = error_criterion
        self._min_information_gain = min_information_gain
        self._max_depth = max_depth
        self._tree = None
        self._trained = False
        self._depth = -1

    def get_tree(self):
        return self._tree

    def get_depth(self):
        return self._depth

    def train(self, x, y):
        self._tree = self._create_tree(x, y)
        self._trained = True

    def predict(self, x):
        tags_predicted_list = []
        for i in range(x.shape[0]):
            tag = self._predict_sample(x[i], self._tree)
            tags_predicted_list.append(tag)
        return tags_predicted_list

    def _predict_sample(self, x, node):
        if type(node) != dict:
            return int(node)

        if float(x[int(node['Feature'])]) <= float(node['Threshold']):
            new_node = node['node']['left']
        else:
            new_node = node['node']['right']
        return self._predict_sample(x, new_node)

    def evaluate(self, x, y):
        if not self._trained:
            print("Error : Please train before evaluating!")
        else:
            predicted = self.predict(x)
            error = 0
            n = x.shape[0]
            for sample in range(n):
                if predicted[sample] != y[sample]:
                    error += 1
            return error/n

    def _partition(self, array, thr):
        return {'left': (array <= thr).nonzero(), 'right': (array > thr).nonzero()}

    def _is_uniform(self, y):
        return y.all() or not y.any()

    def _create_tree(self, x, y):
        self._depth += 1
        # if the set is uniform stop splitting
        if len(y) == 0:
            return y
        if self._is_uniform(y) or self._depth > self._max_depth:
            tag = round(np.mean(y))
            return tag

        # find the split with the max information gain
        thr_list = []
        gain = []
        sorted_x = np.sort(x, axis=0)
        for feature in range(x.shape[1]):
            for i in range(x.shape[0] - 1):
                thr = 0.5 * (sorted_x[i, feature] + sorted_x[i + 1, feature])
                thr_list = np.append(thr_list, thr)
                gain = np.append(gain, self._information_gain(y, x[:, feature], thr))

        mat = gain.reshape(x.shape[1], x.shape[0] - 1).T
        thr_mat = thr_list.reshape(x.shape[1], x.shape[0] - 1).T
        sample, selected_feature = np.unravel_index(mat.argmax(), mat.shape)
        selected_thr = thr_mat[sample, selected_feature]

        if np.all(gain < self._min_information_gain):
            tag = round(np.mean(y))
            return tag
        tree = {'node': {}}
        tree['Threshold'] = selected_thr
        tree['Feature'] = selected_feature

        # split the array using the selected split
        sets = self._partition(x[:, selected_feature], selected_thr)
        for side, vals in sets.items():
            if side == "left":
                x_subset = x[x[:, selected_feature] <= selected_thr]
                y_subset = y[x[:, selected_feature] <= selected_thr]
                left_subtree = self._create_tree(x_subset, y_subset)
                tree['node'][side] = left_subtree

            elif side == "right":
                x_subset = x[x[:, selected_feature] > selected_thr]
                y_subset = y[x[:, selected_feature] > selected_thr]
                right_subtree = self._create_tree(x_subset, y_subset)
                tree['node'][side] = right_subtree
        return tree

    def _misclassification_criterion(self, s):
        if s.shape[0] == 0:
            return 0
        max_p = 0
        tags, counts = np.unique(s, return_counts=True)
        probabilities = counts.astype('float') / len(s)
        for p in probabilities:
            if p > max_p:
                max_p = p
        res = 1 - max_p

        return res

    def _gini_criterion(self , s):
        res = 0
        tags, counts = np.unique(s, return_counts=True)
        probabilities = counts.astype('float') / len(s)
        for p in probabilities:
            if p != 0.0:
                res += p * (1 - p)

        return res

    def _entropy_criterion(self, s):
        res = 0
        # tags include all the tags on the set s , counts include the number of times they appeared in the set
        tags, counts = np.unique(s, return_counts=True)
        probabilities = counts.astype('float')/len(s)
        for p in probabilities:
            if p != 0.0:
                res -= p * np.log2(p)
        return res

    def _information_gain(self, y, x, thr):
        if self._error_criterion == "entropy":
            res = self._entropy_criterion(y)
            counts = np.array([x[x <= thr].shape[0], x[x > thr].shape[0]])
            probabilities = counts.astype('float')/len(x)
            res -= (probabilities[0] * self._entropy_criterion(y[x <= thr])\
                   + probabilities[1] * self._entropy_criterion(y[x > thr]))

            return res

        elif self._error_criterion == "gini":
            res = self._gini_criterion(y)
            counts = np.array([x[x <= thr].shape[0], x[x > thr].shape[0]])
            probabilities = counts.astype('float')/len(x)
            res -= (probabilities[0] * self._gini_criterion(y[x <= thr])\
                   + probabilities[1] * self._gini_criterion(y[x > thr]))

            return res

        elif self._error_criterion == "misclassification":
            res = self._misclassification_criterion(y)
            counts = np.array([x[x <= thr].shape[0], x[x > thr].shape[0]])
            probabilities = counts.astype('float')/len(x)
            res -= (probabilities[0] * self._misclassification_criterion(y[x <= thr])\
                   + probabilities[1] * self._misclassification_criterion(y[x > thr]))

            return res

        else:
            print("Please Enter a Valid Criterion: 'entropy', 'gini', 'misclassification'")



