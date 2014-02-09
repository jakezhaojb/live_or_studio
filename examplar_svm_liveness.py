# This project is designed for experimenting an Ensambled Examplar-SVM model 
# on the liveness problem. We will train a large amount of Examplar SVMs, 
# representing music of both versions, "live" and "studio".

# Written by Junbo Zhao, at Douban Inc., 1/20/2014
# This model is highly compatible with paralized systems. 
# Thus it is recommendaed to use some tools like hadoop, spark and dpark.
# In this python based project, dpark is much prefered.

##======================================================================

import sys
sys.path.append('../libsvm-3.17/python')
import random
import numpy as np
import matplotlib.pyplot as plt
from dpark import DparkContext
from svmutil import svm_read_problem, svm_train, svm_predict

 
def evaluation(test_la, pred_la):
    [FN, FP, TN, TP] = ['FN', 'FP', 'TN', 'TP']
    cnt = {FN: 0, FP: 0, TN: 0, TP: 0}
    for (t_la, p_la) in zip(test_la, pred_la):
        if t_la == p_la:
            if p_la == 1.0:
                cnt[TP] = cnt.get(TP) + 1
            else:
                cnt[TN] = cnt.get(TN) + 1
        else:
            if p_la == 1.0:
                cnt[FP] = cnt.get(FP) + 1
            else:
                cnt[FN] = cnt.get(FN) + 1
    pos_rate = 1. * cnt[TP] / (cnt[TP] + cnt[FP])
    neg_rate = 1. * cnt[TN] / (cnt[TN] + cnt[FN])
    rate = (pos_rate + neg_rate) / 2.0
    return rate, pos_rate, neg_rate


def find(self, value):
    if not isinstance(self, (np.ndarray, list)):
        print("Wrong input parameters")
        return
    length = len(self)
    ind = []
    for i in range(length):
        if self[i] == value:
            ind.append(i)
    return ind


def main(argv):
    # Dpark initialize
    dpark = DparkContext()

    # number of the training and testing set
    num_train = 6000
    num_test = 6000

    # Loading the dataset
    data = svm_read_problem('echo_liveness.01.libsvm')
    y, x = data

    # Preparing training and testing data
    if len(x) != len(y):
        print("The labels and features are not accorded!")
        sys.exit()
    
    x_live = [x[i] for i in find(y, 1.0)]
    x_stu = [x[i] for i in find(y, 0.0)]
    n_live = len(x_live)
    n_stu = len(x_stu)
    ind_live = range(n_live)
    ind_stu = range(n_stu)
    random.shuffle(ind_live)
    random.shuffle(ind_stu)

    x_te = [x_live[i] for i in ind_live[num_train : num_test + num_train]] + \
        [x_stu[i] for i in ind_stu[num_train : num_test + num_train]]
    y_te = [1.0] * len(ind_live[num_train : num_test + num_train]) + \
        [-1.0]*len(ind_stu[num_train : num_test + num_train])
    x_tr = [x_live[i] for i in ind_live[:num_train]] + \
        [x_stu[i] for i in ind_stu[:num_train]]
    y_tr = [1.0]*num_train + [-1.0]*num_train

    # dpark version
    def map_iter(i):
        y_tr_examplar = [-1.0] * len(y_tr)
        y_tr_examplar[i] = 1.0
        # opt = '-t 0 -w1 ' + str(len(y_tr)) + ' -w-1 1 -b 1 -q'
        # It is suggested in Efros' paper that:
        # C1 0.5, C2 0.01
        opt = '-t 0 -w1 0.5 -w-1 0.01 -b 1 -q'
        m = svm_train(y_tr_examplar, list(x_tr), opt)
        p_label, p_acc, p_val = svm_predict(y_te, x_te, m, '-b 1 -q')
        p_val = np.array(p_val)
        # p_val = np.delete(p_val,1,1)  # shape = (N, 1)
        p_val = p_val[:, 0]  # shape = (N, )
        return p_val

    p_vals = dpark.makeRDD(
        range(len(y_tr))
    ).map(
        map_iter
    ).collect()

    val = np.array(p_vals).T

    # for-loop version
    '''
    # Examplar SVM Training
    ensemble_model = []
    # DPark

    for i in range(len(y_tr)):
        y_tr_examplar = [-1.0] * len(y_tr)
        y_tr_examplar[i] = 1.0;
        #opt = '-t 0 -w1 ' + str(len(y_tr)) + ' -w-1 1 -b 1 -q'
        # It is suggested in Efros' paper that:
        # C1 0.5, C2 0.01
        opt = '-t 0 -w1 0.5 -w-1 0.01 -b 1 -q'
        m = svm_train(y_tr_examplar, x_tr, opt)
        ensemble_model.append(m)
        print("The %s-th examplar SVM has been trained" %i)

    # Calibaration, to be updated
    # Since we adopt the probability estimation model of LIB_SVM, Calibrating seems unnecessary

    # Ensembly Classify
    val = np.zeros((len(y_te),1))
    for m in ensemble_model:
        p_label, p_acc, p_val = svm_predict(y_te, x_te, m, '-b 1 -q')
        p_val = np.array(p_val)
        p_val = np.delete(p_val,1, 1)
        val = np.hstack((val, p_val))
    if val.shape[1] != len(y_tr) + 1:
        print "Chaos!"
    val = np.delete(val,0,1)
    print 'val.shape =', val.shape
    '''
    
    # KNN
    k = num_train / 8
    sorted_index = val.argsort(axis=1)
    sorted_index = sorted_index.T[::-1].T
    p_label = []
    for index in sorted_index:
        nearest_samples = []
        for sample_index in index[:k]:
            nearest_samples.append(y_tr[sample_index])
        n,bins,dummy = plt.hist(nearest_samples, 2, normed=1, 
                                facecolor='r', alpha=0.75)
        if n[0] > n[1]:
            p_label.append(-1.0)
        else:
            p_label.append(1.0)

    # evaluation
    rate, pos_rate, neg_rate = evaluation(y_te, p_label)

    print("The Examplar SVM framework achieves a precision of %f" %rate)


if __name__ == '__main__':
    main(sys.argv)
