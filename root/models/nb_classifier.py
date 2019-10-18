import numpy as np
from scipy import sparse
import math

class SparseNaiveBayes:

    split = dict()
    thetas = dict()
    class_probs = dict()
    l_funcs = dict()
    x_train = None
    y_train = None

    def __init__(self):
        return

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.split_x()
        self.get_class_probs()
        self.get_thetas()

    # Assumes that x_test is a sparse matrix
    def predict(self, x_test):
        all_pred = []
        for i in range(x_test.shape[0]):
            cur_example = x_test[i]
            pred = self.predict_y(cur_example)
            all_pred.append(pred)
        return np.ndarray(all_pred)

    # Split x into separate sparse matricies based on their class
    def split_x(self):
        x_split = dict()
        # Iterate example by example
        for i in range(0, len(self.y_train)):
            cls = self.y_train[i]
            example = self.x_train[i]
            if cls in x_split:  # Stack onto existing split
                x_split[cls] = sparse.vstack([x_split[cls], example])
            else:  # Create new key for split
                x_split[cls] = sparse.csr_matrix(example)

        self.split = x_split
        return x_split

    def get_class_probs(self):
        cp = dict()
        for cls in self.split:
            cp[cls] = self.split[cls].shape[0]/self.x_train.shape[0]

        self.class_probs = cp
        return cp

    def get_thetas(self):
        t = dict()
        # For each feature in each class
        for cls in self.split:
            cur_split = self.split[cls]
            non_zero = cur_split.nonzero()[1]
            # Count occurences of non-zeros in columns
            unique, counts = np.unique(non_zero, return_counts=True)
            occurrences = dict(zip(unique, counts))
            t[cls] = [0] * cur_split.shape[1]
            for j in range(0, cur_split.shape[1]):
                occ = 0
                if j in occurrences:
                    occ = occurrences[j]
                t[cls][j] = (occ+1)/(cur_split.shape[0]+2)

        self.thetas = t
        return t

    # Returns likelihood of a given class
    def get_class_likelihood(self, x, cls):
        cur_thetas = self.thetas[cls]
        feature_likelihood = 0
        for j in range(0, len(cur_thetas)):
            feature_likelihood += x[0, j]*math.log(cur_thetas[j]) + (1-x[0, j])*math.log(1-cur_thetas[j])
        likelihood = feature_likelihood + math.log(self.class_probs[cls])
        return likelihood

    def predict_y(self, x):
        likelihoods = dict()
        for cls in self.thetas:
            likelihoods[cls] = self.get_class_likelihood(x, cls)
            print(cls, likelihoods[cls])
        max_likelihood = -math.inf
        best_cls = None
        for cls in likelihoods:
            if likelihoods[cls] > max_likelihood:
                best_cls = cls
                max_likelihood = likelihoods[cls]
        return best_cls


# Testing
model = SparseNaiveBayes()
x = sparse.load_npz('../data/xtrainbin.npz')
y = np.load('../data/y_train.npy')

model.fit(x, y)

print('Fitting Complete.')

pred = model.predict_y(x[1])
print(pred)