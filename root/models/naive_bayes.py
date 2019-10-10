import pandas as pd
import numpy as np
from operator import add
import math

# Returns a list of conditonal probabilities for every x_i given a y_i
def cond_prob(dataset, class_val, class_var):
    # Drop rows in other class variables
    dataset = dataset.drop(dataset[dataset[class_var] != class_val].index)
    # Drop class variable
    dataset = dataset.drop(class_var, axis='columns')
    # Count proportion of 1's for each column
    probs = {}
    for col in dataset.columns:
        probs.append(dataset[col].value_counts()[1.0]/len(dataset))
    return probs


# Returns dictionary on y_i of the conditonal probabilities for each x_i
def get_thetas(dataset, class_var):
    thetas = {}
    for y_i in dataset[class_var].unique():
        thetas[y_i] = cond_prob(dataset, y_i, class_var)

    return thetas


# Returns the summed theta's for classes other than the indicated class value
def get_other_thetas(theta_dict, class_val):
    other_thetas = [0] * len(theta_dict[next(iter(theta_dict))])
    for c in theta_dict:
        if c != class_val:
                other_thetas = list(map(add, theta_dict[c], other_thetas))
    return other_thetas


# Return the raw probability of each class
def get_class_probs(dataset, class_var):
    probs = {}
    class_counts = dataset[class_var].value_count()
    for y_i in dataset[class_var].unique():
        probs[y_i] = class_counts[y_i]/len(dataset)

    return probs


# Return a function for the log likelihood of a given class
def get_log_likelihood(dataset, class_val, class_var):
    class_probs = get_class_probs
    thetas = get_thetas(dataset, class_var)
    mytheta = thetas[class_val]
    other_thetas = get_other_thetas(thetas, class_val)

    def lfunc(x):  # Likelihood function that takes in a validation/test datapoint
        # Term 1
        likelihood = math.log(class_probs[class_val]/(1-class_probs[class_val]))
        # Summation Term
        for j in range(0, len(x)):
            likelihood += x[j] * math.log(mytheta[j]/other_thetas[j]) + (1-x[j]) * math.log(1-mytheta[j]/1-other_thetas[j])
        return likelihood

    return lfunc


