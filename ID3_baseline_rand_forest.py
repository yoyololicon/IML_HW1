from random import shuffle
import random_forest
from collections import Counter
import ID3_baseline
import copy

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def evaluate(lb, data, frst):
    TP = FP = FN = TN = 0
    for item in data:
        predicts = []
        for t in frst:
            predicts.append(ID3_baseline.classify(t, item[:4]))
        my_predict = Most_Common(predicts)
        if item[4] == lb:
            if my_predict == lb:
                TP+=1
            else:
                FN+=1
        else:
            if my_predict == lb:
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
    data = ID3_baseline.get_iris_data('bezdekIris.data')
    feature_div = ID3_baseline.make_boundary(data)

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
        forest = []
        for j in range(K):
            if j != i:
                train+=kfold_data[j]
        for j in range(50):
            forest.append(ID3_baseline.ID3(random_forest.rand_feature(ID3_baseline.features, copy.deepcopy(feature_div), 3), random_forest.rand_data(train, int(len(train)*0.35))))
        tp = 0
        for j in test:
            predicts = []
            for t in forest:
                predicts.append(ID3_baseline.classify(t, j[:4])) #<< next time start here!
            if j[4] == Most_Common(predicts):
                tp+=1
        total_accuracy.append(float(tp)/len(test))
        
        for k in range(3):
            p, r = evaluate(ID3_baseline.labels[k], test, forest)
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

    for i in range(K):
        test = kfold_data[i]
        train = []
        forest = []
        for j in range(K):
            if j != i:
                train += kfold_data[j]
        for j in range(50):
            forest.append(
                ID3_baseline.ID3(random_forest.rand_feature(ID3_baseline.features, copy.deepcopy(feature_div), 3),
                                   random_forest.rand_data(train, int(len(train) * 0.35))))
        tp = 0
        for j in test:
            predicts = []
            for t in forest:
                predicts.append(ID3_baseline.classify(t, j[:4]))  # << next time start here!
            if j[4] == Most_Common(predicts):
                tp += 1
        total_accuracy.append(float(tp) / len(test))

        for k in range(3):
            p, r = evaluate(ID3_baseline.labels[k], test, forest)
            precision[k].append(p)
            recall[k].append(r)

    score = 1.5*sum(total_accuracy)/K + sum(sum(x)/K for x in precision) + sum(sum(x)/K for x in recall)
    return score