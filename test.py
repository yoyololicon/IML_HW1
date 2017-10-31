import ID3_baseline
import ID3_all_feature
import ID3_multib
import ID3_multib_all_feature
import ID3_all_feature_rand_forest
import ID3_multib_rand_forest

scores = []
for i in range(50):
    tmp = ID3_multib_rand_forest.compute_average_score()
    scores.append(tmp)
    print tmp

print 'average', sum(scores)/50