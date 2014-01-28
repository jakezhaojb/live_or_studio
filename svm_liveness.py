#	This project aims at a classification problem - live and studio.
#   We simply adopt LIBSVM in this script, with a module of chasing the best parameters.
#	Designed by Junbo Zhao, at Douban Inc., 1/16/2014

import numpy as np
import random
import os
import sys
sys.path.append(r'../libsvm-3.17/python')
from dpark import DparkContext
from svmutil import *

#
def find(self,value):
    if not isinstance(self, (np.ndarray, list)):
        print("Wrong input parameters")
        return
    length = len(self)
    ind = []
    for i in range(length):
        if self[i] == value:
            ind.append(i)
    return ind

# number of training and testing datasets
if len(sys.argv)!=3:
	print "Wrong input, you should indicate your training and testing sample numbers"
	sys.exit(0)
else:
	num_train = int(sys.argv[1])
	num_test = int(sys.argv[2])


def main(argv):
    # Loading the dataset
    data = svm_read_problem('echo_liveness.01.libsvm')
    y, x = data
    del data

#    num_train = 990
#    num_test = 900

    # Preparing training and testing data
    if len(x) != len(y):
        print("Please examine the data set, for the labels and features are not accorded!")
        sys.exit()
    # generating random training and testing set, to yield the ability of classifier more accurately.
    x_live = [x[i] for i in find(y, 1.0)]
    x_stu = [x[i] for i in find(y, 0.0)]
    n_live = len(x_live)
    n_stu =  len(x_stu)
    ind_live = range(n_live)
    ind_stu = range(n_stu)
    random.shuffle(ind_live)
    random.shuffle(ind_stu)

    x_te = [x_live[i] for i in ind_live[num_train:num_test+num_train]] + [x_stu[i] for i in ind_stu[num_train:num_test+num_train]]
    y_te = [1.0]*len(ind_live[num_train:num_test+num_train]) + [-1.0]*len(ind_stu[num_train:num_test+num_train])
    x_tr = [x_live[i] for i in ind_live[:num_train]] + [x_stu[i] for i in ind_stu[:num_train]]
    y_tr = [1.0]*num_train + [-1.0]*num_train

    # SVM and a 10-fold Cross Validation chasing the best parameters.
    # gamma and c_reg are constructing a parameter grid!
    # Now we focus on the parameters.
    # gamma, c, -v
    
    # for-loop version
    '''
    gamma = np.arange(.01,20,.04)
    c_reg = np.arange(.01,20,.04)
    opt = []
    best_para = {'gamma': 0, 'c': 0, 'precision': 0}
    for g in gamma:
        for c in c_reg:
            opt = '-g '+ str(g) +' -c ' + str(c) + ' -v 10 -q'
            pre = svm_train(y_tr,x_tr,opt)
            if pre > best_para.get('precision'):
                best_para['gamma'] = g
                best_para['c'] = c
                best_para['precision'] = pre 
    best_opt = '-g '+ str(best_para.get('gamma')) +' -c ' + str(best_para.get('c')) + ' -q'
    m = svm_train(y_tr, x_tr, best_opt)
    p_label, p_acc, p_val = svm_predict(y_te, x_te, m, '-q')
    '''

    # dpark version
    dpark = DparkContext()
    gamma = np.arange(.01,5,.08)
    c_reg = np.arange(.01,5,.08)
    opt = []
    for g in gamma:
        for c in c_reg:
            opt.append('-g '+ str(g) +' -c ' + str(c) + ' -v 10 -q')

    def map_iter(i):
        pre = svm_train(y_tr,list(x_tr),opt[i])
        return pre

    #pres = dpark.makeRDD(range(len(opt)),100).map(map_iter).collect()
    pres = dpark.makeRDD(range(len(opt))).map(map_iter).collect()
    pres = np.array(pres)
    best_opt_ind = pres.argsort()
    best_opt = opt[best_opt_ind[-1]]

    best_opt = best_opt[:best_opt.find('-v')-1]
    m = svm_train(y_tr, x_tr, best_opt)
    p_label, p_acc, p_val = svm_predict(y_te, x_te, m, '-q')

    print 'This solely SVM framework achieves a precision of %f' %p_acc[0]

if __name__ == '__main__':
    main(sys.argv)
