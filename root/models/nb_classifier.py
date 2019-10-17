import numpy as np
from scipy import sparse

class SparseNaiveBayes:

    def __init__(self):
        return

    # Split x into separate sparse matricies based on their class
    @staticmethod
    def split_x(x_train, y_train):
        x_split = dict()
        # Iterate example by example
        for i in range(0, len(y_train)):
            cls = y_train[i]
            example = x_train[i]
            if cls in x_split:  # Stack onto existing split
                x_split[cls] = sparse.vstack([x_split[cls], example])
            else:  # Create new key for split
                x_split[cls] = sparse.csr_matrix(example)
        return x_split

    @staticmethod
    def get_class_probs(x_train, x_split):
        class_probs = dict()
        for cls in x_split:
            class_probs[cls] = x_split[cls].shape[0]/x_train.shape[0]
        return class_probs

    @staticmethod
    def get_thetas(x_split):
        thetas = dict()
        # For each feature in each class
        for cls in x_split:
            cur_split = x_split[cls]
            non_zero = cur_split.nonzero()[1]
            # Count occurences of non-zeros in columns
            unique, counts = np.unique(non_zero, return_counts=True)
            occurrences = dict(zip(unique, counts))
            thetas[cls] = [0] * cur_split.shape[1]
            for key in occurrences:
                thetas[cls][key] = occurrences[key]

        return thetas


# Testing
model = SparseNaiveBayes()
x = sparse.load_npz('../data/xtrainbin.npz')
y = np.load('../data/y_train.npy')


# Splitting
split = model.split_x(x, y)

# Class Probs
class_probs = model.get_class_probs(x, split)
print(class_probs)

# Thetas
thetas = model.get_thetas(split)
print(type(thetas))