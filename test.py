import ID3_baseline
import ID3_allFeature
import ID3_multib
import ID3_multib_allFeature
import ID3_allFeature_rand_forest
import ID3_multib_rand_forest
import ID3_baseline_rand_forest

scores = []
for i in range(50):
    tmp = ID3_baseline_rand_forest.compute_average_score()
    scores.append(tmp)
    print tmp

print 'average', sum(scores)/50