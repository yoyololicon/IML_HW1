from itertools import combinations
from random import shuffle
import ID3_baseline
import copy

class Node:
  def __init__(self, leaf=None):
    self.feature = None
    self.label = leaf
    self.children = []
    
def remainder(d, D, k):
    M = len(D)
    rem_min = float('inf')
    ft = [0, 0]
    
    for f in d:
        idx = ID3_baseline.features.index(f[0])
        comb = combinations(range(len(f[1])), k)
        #print f
        for boundary in comb:
            b = []
            for q in boundary:
                b.append(f[1][q])
            e = []
            dd = []
            dd.append([x for x in D if x[idx] < b[0]])
            e.append(ID3_baseline.entropy(dd[0]))
            for i in range(k-1):
                dd.append([x for x in D if b[i] <= x[idx] < b[i+1]])
                e.append(ID3_baseline.entropy(dd[i + 1]))
            dd.append([x for x in D if b[k-1] <= x[idx]])
            e.append(ID3_baseline.entropy(dd[k]))
            
            rem = 0
            for i in range(k+1):
                rem+=float(len(dd[i]))/M*e[i]

            if rem < rem_min:
                rem_min = rem
                ft[0] = idx
                ft[1] = b
    
    return ft, rem_min
            
def ID3(d, D, k):
    #case 1
    if all(D[0][4] == x[4] for x in D):
        return Node(leaf=D[0][4])
    #case 2
    elif len(d) == 0:
        return Node(leaf=ID3_baseline.get_major_feature(D))
    else:
        parti, rem = remainder(d, D, k)
        dd = []
        dd.append([x for x in D if x[parti[0]] < parti[1][0]])
        for i in range(k-1):
            dd.append([x for x in D if parti[1][i] <= x[parti[0]] < parti[1][i+1]])
        dd.append([x for x in D if parti[1][k-1] <= x[parti[0]]])
        
        new_n = Node()
        new_n.feature = parti
        for i in d:
            if i[0] == ID3_baseline.features[parti[0]]:
                d.remove(i)
                break
        #case 3
        for sub_D in dd:
            if len(sub_D):
                new_n.children.append(ID3(d, sub_D, k))
            else:
                new_n.children.append(Node(leaf=ID3_baseline.get_major_feature(D)))

        return new_n
        
def classify(tree, x):
    if tree.label:
        return tree.label
    elif x[tree.feature[0]] < tree.feature[1][0]:
        return classify(tree.children[0], x)
    elif x[tree.feature[0]] >= tree.feature[1][len(tree.feature[1])-1]:
        return classify(tree.children[len(tree.feature[1])], x)
    else:
        for i in range(len(tree.feature[1])-1):
            if tree.feature[1][i] <= x[tree.feature[0]] < tree.feature[1][i+1]:
                return classify(tree.children[i+1], x)
    
def evaluate(lb, data, rt):
    TP = FP = FN = TN = 0
    for item in data:
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
    data = ID3_baseline.get_iris_data('bezdekIris.data')
    feature_div = ID3_baseline.make_boundary(data)
    
    shuffle(data)
    K = 5
    stepsize = len(data)/K
    kfold_data = [data[i:i + stepsize] for i in range(0, len(data), stepsize)]
    total_accuracy = []
    precision = [[], [], []]
    recall = [[], [], []]
    
    cut = 2
    
    for i in range(K):
        test = kfold_data[i]
        train = []
        for j in range(K):
            if j != i:
                train+=kfold_data[j]
        root = ID3(copy.deepcopy(feature_div), train, cut)
        tp = 0
        for j in test:
            if j[4] == classify(root, j[:4]):
                tp+=1
        total_accuracy.append(float(tp)/len(test))
        
        for k in range(3):
            p, r = evaluate(ID3_baseline.labels[k], test, root)
            precision[k].append(p)
            recall[k].append(r)
    
    print sum(total_accuracy)/K
    print sum(precision[0])/K, sum(recall[0])/K
    print sum(precision[1])/K, sum(recall[1])/K
    print sum(precision[2])/K, sum(recall[2])/K

def compute_average_score():
    data = ID3_baseline.get_iris_data('bezdekIris.data')
    feature_div = ID3_baseline.make_boundary(data)

    shuffle(data)
    K = 5
    stepsize = len(data) / K
    kfold_data = [data[i:i + stepsize] for i in range(0, len(data), stepsize)]
    total_accuracy = []
    precision = [[], [], []]
    recall = [[], [], []]

    cut = 2

    for i in range(K):
        test = kfold_data[i]
        train = []
        for j in range(K):
            if j != i:
                train += kfold_data[j]
        root = ID3(copy.deepcopy(feature_div), train, cut)
        tp = 0
        for j in test:
            if j[4] == classify(root, j[:4]):
                tp += 1
        total_accuracy.append(float(tp) / len(test))

        for k in range(3):
            p, r = evaluate(ID3_baseline.labels[k], test, root)
            precision[k].append(p)
            recall[k].append(r)

    score = 1.5*sum(total_accuracy)/K + sum(sum(x)/K for x in precision) + sum(sum(x)/K for x in recall)
    return score