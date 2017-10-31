#!/bin/bash
# this will execute the ID3 algorithm for bezdekIris.data with K-fold and random forest

PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
export PATH
python2 ID3_allFeature_rand_forest.py
exit 0
