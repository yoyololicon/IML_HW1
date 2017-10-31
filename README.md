# IML_HW1_report
ID3	algorithm

## 開發環境和語言
* ubuntu 16.04 LTS
* python2.7.12(using Pycharm 2017.2.3)
* 除了standard modules之外沒有使用其他外部package

## 實做細節
總共寫了六種版本來實做ID3演算法

### ID3_baseline
最一開始的版本，每個node只有兩個child，大於或小於該node的feature分割點。
每個feature用掉一個分割點後就會捨棄，所以tree最高只會有四層。

### ID3_all_feature
基於baseline，但是feature會用掉所有分割點後才會捨棄。

### ID3_all_feature_rand_forest
將ID3_all_feature加上random forest的功能。總共有10個tree，feature選四次，重複的替除，資料則是取training set大小的3/4次(120*0.75 = 90)
包含重複的.

### ID3_multib
在baseline裡，只會有兩個child，不是大於等於該node的判斷條件就是小於。於是我想如果可以分割成更多塊會不會好點？
此版本就是把child變成三個的結果，需要兩個分割點來做判斷。

### IDE_multib_all_feature
和ID3_all_feature一樣會用掉所有分割點後才會捨棄feature。

### IDE_multib_rand_forest
將IDE3加上random forest功能。tree和資料的建構方式同ID3_all_feature_rand_forest。
