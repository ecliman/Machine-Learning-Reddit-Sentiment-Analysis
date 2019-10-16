import numpy as np

class NaiveBayes:
    l_funcs: {} # likelihood functions
    x_train = None
    y_train = None
    x_split = {}
    class_probs = None

    # Split x into separate ndarrays based on their class
    def __split_x(self, x_train, y_train):
        x_split = dict()
        for row in range(0, x_train.shape[0]):
            key = y_train[row]
            if key in x_split:
                x_split[key].append(x_train[row])
            else:
                x_split[key] = [x_train[row]]
        for key in x_split:
            x_split[key] = np.asarray(x_split[key])

        self.x_split = x_split

    @staticmethod
    def __cond_prob(x_split, class_val):
        probs = list()
        split = x_split[class_val]
        for col in range(0, split.shape[1]):
            count = 0
            for row in range(0, split.shape[0]):
                if split[row][col] > 0:
                    count += 1
            probs.append(count)
        for p in range(0, len(probs)):
            probs[p] = probs[p] / split.shape[0]
        return probs

