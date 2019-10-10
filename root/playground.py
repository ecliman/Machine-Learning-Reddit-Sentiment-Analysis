import pandas as pd
from operator import add
data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data,columns=['Name','Age'],dtype=float)
# print(df['Age'].value_counts()[13.0])

tdict = {
    1: [1, 1, 1],
    2: [1, 1, 1],
    3: [1, 1, 1],
    4: [1, 1, 1],

}

print(next(iter(tdict)))

def get_other_thetas(theta_dict, class_val):
    other_thetas = [0] * len(theta_dict[next(iter(theta_dict))])
    for c in theta_dict:
        if c != class_val:
            other_thetas = list(map(add, theta_dict[c], other_thetas))

    return other_thetas

print(get_other_thetas(tdict, 1))
