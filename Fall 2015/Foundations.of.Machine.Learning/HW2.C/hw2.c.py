__author__ = 'yunlinz'

import svm
import svmutil
import sklearn.cross_validation
import numpy
import matplotlib.pyplot as plt
import time
import ctypes

# load data
y_train, x_train = svmutil.svm_read_problem('splice_noise_train.txt')
y_test, x_test = svmutil.svm_read_problem('splice_noise_test.txt')

# scale x_train to [-1,1]
x_train_sum = {}
x_train_min = {}
x_train_max = {}
x_train_rng = {}
x_train_scaled = []
x_test_scaled = []

for d in x_train:
    for i in range(1, 60, 1):
        if i in x_train_min:
            if d[i] < x_train_min[i]:
                x_train_min[i] = d[i]
        else:
            x_train_min[i] = d[i]
        if i in x_train_max:
            if d[i] > x_train_max[i]:
                x_train_max[i] = d[i]
        else:
            x_train_max[i] = d[i]
        if i in x_train_sum:
            x_train_sum[i] += d[i]
        else:
            x_train_sum[i] = d[i]

for i in range(1, 60, 1):
    x_train_rng[i] = x_train_max[i] - x_train_min[i]

for d in x_train:
    s = {}
    for i in range(1, 60, 1):
        s[i] = (d[i] - x_train_sum[i] / y_train.__len__()) / x_train_rng[i]
    x_train_scaled.append(s)

for d in x_test:
    s = {}
    for i in range(1, 60, 1):
        s[i] = (d[i] - x_train_sum[i] / y_train.__len__()) / x_train_rng[i]
    x_test_scaled.append(s)

folds = 10
kfold_iterator = sklearn.cross_validation.KFold(y_train.__len__(), folds, shuffle=True)

# polynomial kernel d=1,3,5
# plot average cross-validation error as function of C from 5^-k to 5^k
k = 7
k_range = range(-k, k + 1)
d = [1, 3, 5]

acc = numpy.zeros([k_range.__len__(), folds, d.__len__()])
nSV = numpy.zeros([k_range.__len__(), folds, d.__len__()])
ctm = numpy.zeros([k_range.__len__(), folds, d.__len__()])

mean = numpy.array([])
for degree_idx in range(0, d.__len__()):
    for k_range_idx in range(0, k_range.__len__()):
        run = 0
        print "Polynomial degree: " + d[degree_idx].__str__() + " C: " + (5**k_range[k_range_idx]).__str__()
        for train, validate in kfold_iterator:
            x_train_scaled_train, y_train_train = map(x_train_scaled.__getitem__, train), map(y_train.__getitem__, train)
            x_train_scaled_validate, y_train_validate = map(x_train_scaled.__getitem__, validate), map(y_train.__getitem__, validate)
            start = time.clock()
            model = svmutil.svm_train(y_train_train, x_train_scaled_train, ["-q", "-t", "1",
                                                                            "-d", d[degree_idx],
                                                                            "-c", (5**k_range[k_range_idx]),
                                                                            "-h", "1",
                                                                            "-m", "512"])
            p_label, p_acc, p_val = svmutil.svm_predict(y_train_validate, x_train_scaled_validate, model)
            end = time.clock()
            acc[k_range_idx, run, degree_idx] = 1 - p_acc[0] / 100
            nSV[k_range_idx, run, degree_idx] = model.get_nr_sv()
            ctm[k_range_idx, run, degree_idx] = end - start
            run += 1

numpy.save('c.2.acc', acc)
numpy.save('c.2.nSV', nSV)
numpy.save('c.2.time', ctm)

acc_mean = numpy.mean(acc, 1)
nSV_mean = numpy.mean(nSV, 1)
ctm_mean = numpy.mean(ctm, 1)
plt.subplot(1, 3, 1)
plt.plot(k_range, acc_mean)
plt.title('Polynomial kernel SVM performance')
plt.ylabel('Error rate')
plt.xlabel('C')

plt.subplot(1, 3, 2)
plt.plot(k_range, nSV_mean)
plt.ylabel('n SV')
plt.xlabel('C')

plt.subplot(1, 3, 3)
plt.plot(k_range, ctm_mean)
plt.ylabel('Training time')
plt.xlabel('C')
plt.show()

