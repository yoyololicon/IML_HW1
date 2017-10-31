import math
from random import shuffle
import copy

labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
features = ['sl', 'sw', 'pl', 'pw']

class bin_Node:
  def __init__(self, leaf=None):
    self.feature = None
    self.label = leaf
    self.children = [None, None]
  
def get_iris_data(file):
    f = open(file)
    raw_data = f.read().split('\n')
    data = []
    for lines in raw_data:
        t = [x for x in lines.split(',')]
        if len(t) > 1:
            for i in range(4):
                t[i] = float(t[i])
            data.append(t)
    return data

def get_major_feature(data):
    counter = []
    data_l = [x[4] for x in data]
    for l in labels:
        counter.append(data_l.count(l))
    return labels[counter.index(max(counter))]
    
def entropy(data):
    if not len(data):
        return 0
    e = 0
    data_l = [l[4] for l in data]
    M = len(data_l)
    for l in labels:
        p = float(data_l.count(l))/M
        if p:
            e+=p*math.log(p, 2)
    return -e
    
def remainder(d, D):
    M = len(D)
    rem_min = float('inf')
    ft = [0, 0]
    
    for f in d:
        idx = features.index(f[0])
        for boundary in f[1]:
            d1 = [x for x in D if x[idx] >= boundary]
            d2 = [x for x in D if x not in d1]
            e1 = entropy(d1)
            e2 = entropy(d2)
            rem = float(len(d1))/M*e1 + float(len(d2))/M*e2

            if rem < rem_min:
                rem_min = rem
                ft[0] = idx
                ft[1] = boundary
    
    return ft, rem_min
            
def ID3(d, D):
    #case 1
    if all(D[0][4] == x[4] for x in D):
        return bin_Node(leaf=D[0][4])
    #case 2
    elif len(d) == 0:
        return bin_Node(leaf=get_major_feature(D))
    else:
        parti, rem = remainder(d, D)
        D1 = [x for x in D if x[parti[0]] < parti[1]]
        D2 = [x for x in D if x not in D1]
        
        new_n = bin_Node()
        new_n.feature = parti
        for i in d:
            if i[0] == features[parti[0]]:
                d.remove(i)
                break
        
        #case 3
        if len(D1):
            new_n.children[0] = ID3(d, D1)
        else:
            new_n.children[0] = bin_Node(leaf=get_major_feature(D))
        if len(D2):
            new_n.children[1] = ID3(d, D2)
        else:
            new_n.children[1] = bin_Node(leaf=get_major_feature(D))

        return new_n
        
def classify(tree, x):
    if tree.label:
        return tree.label
    elif x[tree.feature[0]] < tree.feature[1]:
        return classify(tree.children[0], x)
    else:
        return classify(tree.children[1], x)
        
def make_boundary(data):
    ft = []
    for i in range(4):
        div = []
        data_s = sorted(data,key=lambda d:d[i])
        for j in range(len(data)-1):
            if data_s[j][4] != data_s[j+1][4]:
               div.append((data_s[j][i]+data_s[j+1][i])/2)
        div = sorted(list(set(div)))
        ft.append([features[i], div])
    return ft
    
def evaluate(lb, data, rt):
    TP = FP = FN = TN = 0
    for item in data:
        #print lb, item[4], classify(rt, item[:4])
        if item[4] == lb:
            if classify(rt, item[:4]) == lb:
                TP+=1
            else:
                FN+=1
        else:
            if classify(rt, item[:4]) == lb:
                FP+=1
            else:
                TN+=1
    #print TP, FP, FN, TN
    if TP+FP == 0:
        p = 0
    else:
        p = float(TP)/(TP+FP)
    if TP+FN == 0:
        r = 0
    else:
        r = float(TP)/(TP+FN)
    return p, r
        
if __name__ == '__main__':
    data = get_iris_data('bezdekIris.data')
    feature_div = make_boundary(data)

    shuffle(data)
    K = 5
    stepsize = len(data)/K
    kfold_data = [data[i:i + stepsize] for i in range(0, len(data), stepsize)]
    total_accuracy = []
    precision = [[], [], []]
    recall = [[], [], []]

    for i in range(K):
        test = kfold_data[i]
        train = []
        for j in range(K):
            if j != i:
                train+=kfold_data[j]

        root = ID3(copy.deepcopy(feature_div), train)
        tp = 0
        for j in test:
            if j[4] == classify(root, j[:4]):
                tp+=1
        total_accuracy.append(float(tp)/len(test))

        for k in range(3):
            p, r = evaluate(labels[k], test, root)
            precision[k].append(p)
            recall[k].append(r)

    print sum(total_accuracy)/K
    print sum(precision[0])/K, sum(recall[0])/K
    print sum(precision[1])/K, sum(recall[1])/K
    print sum(precision[2])/K, sum(recall[2])/K
