import pandas as pd
import numpy as np

def naive_bayes(dataset, class_var):


# Returns a list of conditonal probabilities for every x_i given a y_i
def cond_prob(dataset, class_val, class_var):
    # Drop rows in other class variables
    dataset = dataset.drop(dataset[dataset[class_var] != class_val].index)
    # Drop class variable
    dataset = dataset.drop(class_var, axis='columns')
    # Count proportion of 1's for each column
    probs = {}
    for col in dataset.columns:
        probs.append(dataset[col].value_counts()[1]/len(dataset))
    return probs


# Returns dictionary on y_i of the conditonal probabilities for each x_i
def get_thetas(dataset, class_var):
    thetas = {}
    for y_i in dataset[class_var].unique():
        thetas[y_i] = cond_prob(dataset, y_i, class_var)

    return thetas


# Return the probability of each class
def get_class_probs(dataset, class_var):
    probs = {}
    class_counts = dataset[class_var].value_count()
    for y_i in dataset[class_var].unique():
        probs[y_i] = class_counts[y_i]/len(dataset)

    return probs


def log_likelihood(dataset, class_var):
    classes = dataset[class_var].unique()
    class_probs = get_class_probs(dataset, class_var)
    thetas = get_thetas(dataset, class_var)

    # Product of conditional probabilities for each class
    likelihood = None
    # for i in range(0, len(classes)):
    #     prob_yi =