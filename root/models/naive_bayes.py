import pandas as pd
import numpy as np
from operator import add
import math

class NaiveBayes:
    l_funcs: {} # Likelihood functions
    class_var = 'subreddit'
    training_set = None

    # Initializes and trains the model with the dataset
    def __init__(self, training_set, class_var):
        self.class_var = class_var
        self.training_set = training_set
        self.class_probs = self.get_class_probs()
        self.thetas = self.get_thetas(self.training_set)  # Learn thetas
        for y_i in training_set[self.class_var].unique():  # Produce log likelihood of each class variable
            self.l_funcs[y_i] = self.get_log_likelihood(self.training_set, y_i)

    # Predicts the classification for feature vector x
    def predict(self, x):
        return None

    # Return a function for the log likelihood of a given class
    def __get_log_likelihood(self, dataset, class_val):
        mytheta = self.thetas[class_val]
        other_thetas = self.get_other_thetas(self.thetas, class_val)

        def lfunc(x):  # Likelihood function that takes in a validation/test datapoint
            # Term 1
            likelihood = math.log(self.class_probs[class_val]/(1-self.class_probs[class_val]))
            # Summation Term
            for j in range(0, len(x)):
                likelihood += x[j] * math.log(mytheta[j]/other_thetas[j]) + (1-x[j]) * math.log(1-mytheta[j]/1-other_thetas[j])
            return likelihood

        return lfunc

    # Returns a list of conditonal probabilities for every x_i given a y_i
    def __cond_prob(self, dataset, class_val):
        # Drop rows in other class variables
        dataset = dataset.drop(dataset[dataset[self.class_var] != self.class_val].index)
        # Drop class variable
        dataset = dataset.drop(self.class_var, axis='columns')
        # Count proportion of 1's for each column
        probs = {}
        for col in dataset.columns:
            probs.append(dataset[col].value_counts()[1.0]/len(dataset))
        return probs

    # Returns dictionary on y_i of the conditonal probabilities for each x_i
    def __get_thetas(self, dataset):
        thetas = {}
        for y_i in dataset[self.class_var].unique():
            thetas[y_i] = self.cond_prob(dataset, y_i, self.class_var)

        return thetas

    # Returns the summed theta's for classes other than the indicated class value
    @staticmethod
    def __get_other_thetas(theta_dict, class_val):
        other_thetas = [0] * len(theta_dict[next(iter(theta_dict))])
        for c in theta_dict:
            if c != class_val:
                    other_thetas = list(map(add, theta_dict[c], other_thetas))
        return other_thetas

    # Return the raw probability of each class
    def __get_class_probs(self, dataset):
        probs = {}
        class_counts = dataset[self.class_var].value_count()
        for y_i in dataset[self.class_var].unique():
            probs[y_i] = class_counts[y_i]/len(dataset)

        return probs


