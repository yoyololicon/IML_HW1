import random
from itertools import groupby

def rand_feature(labels, ft, k):
    sft = []
    rsft = []
    for i in range(k):
        sft.append(random.choice(labels))
    for i in ft:
        if i[0] in sft:
            rsft.append(i)
    return rsft
    
def rand_data(D, k):
    sD = []
    for i in range(k):
        sD.append(random.choice(D))
    sD_set = set(map(tuple,sD))
    sD = map(list,sD_set)
    return sD