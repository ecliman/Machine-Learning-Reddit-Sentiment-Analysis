import pandas as pd
import numpy as np

Y = 'subreddit'


# partition a dataframe into k parts
def partition(dataframe, k):
    partition_size = len(dataframe) // k
    partitions = []
    for i in range(0, k - 1):
        partitions.append(dataframe.iloc[i * partition_size: (i + 1) * partition_size])

    partitions.append(dataframe.iloc[k - 1 * partition_size:])
    return partitions


def evaluate_acc(dataset, train_and_validate, class_var=Y):
    partitions = partition(dataset, 5)
    # Hold out each set for training once
    accuracies = []
    for i in range(0, len(partitions)):
        holdout = partitions[i]
        training_data = pd.DataFrame(partitions[0])[0:0]
        for j in range(0, len(partitions)):
            if j != i:
                training_data = training_data.append(partitions[j], ignore_index=True)
        accuracies.append(train_and_validate(training_data, holdout, class_var))

    # Average out and return the accuracies for the given features
    return sum(accuracies)/len(accuracies)