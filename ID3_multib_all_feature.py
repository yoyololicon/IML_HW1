from random import shuffle
import ID3_baseline
import ID3_multib
import copy
            
def ID3(d, D, k):
    #case 1
    if all(D[0][4] == x[4] for x in D):
        return ID3_multib.Node(leaf=D[0][4])
    #case 2
    elif len(d) == 0:
        return ID3_multib.Node(leaf=ID3_baseline.get_major_label(D))
    else:
        parti, rem = ID3_multib.remainder(d, D, k)
        dd = []
        dd.append([x for x in D if x[parti[0]] < parti[1][0]])
        for i in range(k-1):
            dd.append([x for x in D if parti[1][i] <= x[parti[0]] < parti[1][i+1]])
        dd.append([x for x in D if parti[1][k-1] <= x[parti[0]]])
        
        new_n = ID3_multib.Node()
        new_n.feature = parti
        for i in range(len(d)):
            if d[i][0] == ID3_baseline.features[parti[0]]:
                for border in parti[1]:
                    d[i][1].remove(border)
                if len(d[i][1]) == 0:
                    del d[i]
                break
        #case 3
        for sub_D in dd:
            if len(sub_D):
                new_n.children.append(ID3(d, sub_D, k))
            else:
                new_n.children.append(ID3_multib.Node(leaf=ID3_baseline.get_major_label(D)))

        return new_n
        
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
    
    cut = 1
    
    for i in range(K):
        test = kfold_data[i]
        train = []
        for j in range(K):
            if j != i:
                train+=kfold_data[j]
        root = ID3(copy.deepcopy(feature_div), train, cut)
        tp = 0
        for j in test:
            if j[4] == ID3_multib.classify(root, j[:4]):
                tp+=1
        total_accuracy.append(float(tp)/len(test))
        
        for k in range(3):
            p, r = ID3_multib.evaluate(ID3_baseline.labels[k], test, root)
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
            if j[4] == ID3_multib.classify(root, j[:4]):
                tp += 1
        total_accuracy.append(float(tp) / len(test))

        for k in range(3):
            p, r = ID3_multib.evaluate(ID3_baseline.labels[k], test, root)
            precision[k].append(p)
            recall[k].append(r)

    score = 1.5*sum(total_accuracy)/K + sum(sum(x)/K for x in precision) + sum(sum(x)/K for x in recall)
    return score