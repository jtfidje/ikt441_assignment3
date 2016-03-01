print ""
import pickle
import sys
import random

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, grid_search

import warnings
warnings.simplefilter("ignore")

''' 
Function that takes a list of even numbers and maps
the data into pairs of two to represent x and y coordinates
'''
def mapTo2D(data):
    retval = [] 
    for i in range(0, len(data), 2):
       x = data[i]
       y = data[i + 1]
       retval.append((x, y))
    return retval

'''
Function that plots datapoints together with 
SVM graph.
'''
def plotSVM(svm, n, title):
    colors = np.array(["g" for i in training_2d_0][train_num:] + 
                      ["r" for i in training_2d_1][train_num:] + 
                      ["b" for i in training_2d_2][train_num:])

    plt.subplot(2, 2, n)
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap = plt.cm.Paired, alpha = 0.8)
    plt.scatter(X[:, 0], X[:, 1], c = colors, cmap = plt.cm.Paired)
    plt.title(title)

'''
Function for testing the SVM's
'''
def testSVM(title, svm, zero, one, two):
    numcorrect = 0
    numwrong = 0
    for correct, testing in ((0, zero),(1, one), (2, two)):
        for d in testing:
            r = svm.predict(d)[0]
            if(r == correct):
                numcorrect += 1
            else:
                numwrong += 1
    
    print title
    print "Correct", numcorrect
    print "Wrong", numwrong
    print numcorrect * 100 / (numcorrect + numwrong), '%', "\n"

    if write_results:
        f = open('results.txt', 'a')
        f.write(title + "\n")
        f.write("Correct: " + str(numcorrect) + "\n")
        f.write("Wrong: " + str(numwrong) + "\n")
        f.write(str(numcorrect * 100 / (numcorrect + numwrong)) + '%' + "\n\n")
        f.close()

'''
Write config vars to file for review
'''
def write_config():
    f = open('results.txt', 'w')
    f.write("Test percentage: " + str(test_perc) + "\n")
    f.write("Training from: " + str(train_num) + "\n")
    f.write("Plotting from: " + str(plot_num) + "\n")
    f.write("Decision Function Shape: " + decision_function_shape + "\n")
    f.write("Polynomial degree: " + str(degree) + "\n")
    f.write("C: " + str(C) + "\n")
    f.write("Gamma: " + str(gamma) + "\n")
    f.write("\n")
    f.close()


# CONFIG VARS #

# Choose wheter to create a results file
write_results = True

# Data file to use
data_file = "matches_per.csv"

# Percentage of data to be used as test data
test_perc = 0.25

# List of classes
classes = [ 1, # Hjemme
            2, # Uavgjort
            3] # Borte

# Where to cut of the data list from the end
cut_off = (-3)

# Where in the traininglist to start training from
train_num = (-10)

# Where in the lists to start plotting from
plot_num = (-100)

# Choose whether to plot the raw data or not
plot_data = False

# Chose whether to train and test in 2D
train_2d = False

# Choose whether to train and test in n-D
train_nd = False

# Choose whether to train using the GridSearchCV class for tuning parameters
train_auto = True

# Choose the decision function shape
decision_function_shape = ["ovr", "ovo"][0]

# Choose the polynomial degree
degree = 2.0

# Choose the C and gamma vars
C = 3.0
gamma = 0.5

# Write config vars to file
if write_results:
    write_config()

# Load data from files
data = [[int(float(i)) for i in i.split(',')] for i in open(data_file, 'r').readlines() if i.strip()]

test_num = int(len(data) * test_perc)

training = data[:test_num]
testing = data[test_num:]

# Extracting and sorting training data into separate lists
training_0 = [i[1:cut_off] for i in training if i[-1] == classes[0]]  # Hjemme
training_1 = [i[1:cut_off] for i in training if i[-1] == classes[1]]  # Uavgjort
training_2 = [i[1:cut_off] for i in training if i[-1] == classes[2]]  # Borte

# Extracting and sorting testing data into separate lists
testing_0 = [i[1:cut_off] for i in testing if i[-1] == classes[0]]    # Hjemme
testing_1 = [i[1:cut_off] for i in testing if i[-1] == classes[1]]    # Uavgjort
testing_2 = [i[1:cut_off] for i in testing if i[-1] == classes[2]]    # Borte



# 2-Dimensions
if train_2d:
    training_2d_0 = []
    training_2d_1 = []
    training_2d_2 = []

    for d in training_0:
        training_2d_0 += mapTo2D(d)

    for d in training_1:
        training_2d_1 += mapTo2D(d)

    for d in training_2:
        training_2d_2 += mapTo2D(d)

    testing_2d_0 = []
    testing_2d_1 = []
    testing_2d_2 = []

    for d in testing_0:
        testing_2d_0 += mapTo2D(d)

    for d in testing_1:
        testing_2d_1 += mapTo2D(d)

    for d in testing_2:
        testing_2d_2 += mapTo2D(d)

    if plot_data:
        print "Creating 2D plot with raw datapoints\n"

        colors = np.array(["g" for x in training_2d_0][plot_num:] + 
                          ["r" for x in training_2d_1][plot_num:] +
                          ["b" for x in training_2d_2][plot_num:])

        overlay = np.array(training_2d_0[plot_num:] + 
                           training_2d_1[plot_num:] + 
                           training_2d_2[plot_num:])

        plt.subplot(2,2,1)
        plt.title("Hjemme")
        plt.plot([i[0] for i in training_2d_0][plot_num:],[i[1] for i in training_2d_0][plot_num:], "o", color = "green")

        plt.subplot(2,2,2)
        plt.title("Uavgjort")
        plt.plot([i[0] for i in training_2d_1][plot_num:],[i[1] for i in training_2d_1][plot_num:], "o", color = "green")

        plt.subplot(2,2,3)
        plt.title("Borte")
        plt.plot([i[0] for i in training_2d_2][plot_num:],[i[1] for i in training_2d_2][plot_num:], "o", color = "green")

        plt.subplot(2,2,4)
        plt.title("Overlay")
        plt.scatter(overlay[:, 0], overlay[:, 1], c = colors, cmap = plt.cm.Paired, marker = "o")

        plt.savefig("fotball_untrained.png")
        plt.clf()

    # Create numpy arrays to use for training. 
    # X contains the features
    # Y contains the classes
    X = np.array(training_2d_0[train_num:] + 
                 training_2d_1[train_num:] + 
                 training_2d_2[train_num:])
    
    Y = np.array([0 for i in training_2d_0][train_num:] + 
                 [1 for i in training_2d_1][train_num:] + 
                 [2 for i in training_2d_2][train_num:])

    print "Starting 2D Training..."

    linear_svm = svm.SVC(decision_function_shape = decision_function_shape, kernel = 'linear', C = C, gamma = gamma).fit(X, Y)
    testSVM("Linear", linear_svm, testing_2d_0, testing_2d_1, testing_2d_2)

    rbf_svm = svm.SVC(decision_function_shape = decision_function_shape, kernel = 'rbf', C = C, gamma = gamma).fit(X, Y)
    testSVM("RBF", rbf_svm, testing_2d_0, testing_2d_1, testing_2d_2)

    sigmoid_svm = svm.SVC(decision_function_shape = decision_function_shape, kernel = 'sigmoid',C = C, gamma = gamma).fit(X, Y)
    testSVM("Sigmoid", sigmoid_svm, testing_2d_0, testing_2d_1, testing_2d_2)

    nu_svm = svm.NuSVC().fit(X, Y)
    testSVM("NuSVC", nu_svm, testing_2d_0, testing_2d_1, testing_2d_2)

    polynomial_svm = svm.SVC(decision_function_shape = decision_function_shape, kernel = 'poly', C = C, gamma = gamma, degree = degree).fit(X,Y)
    testSVM("Polynomial", polynomial_svm, testing_2d_0, testing_2d_1, testing_2d_2)

    # Plot the different SVMs together with the data
    if plot_data:
        print "Creating 2D plot with trained SVMs"
        h = 0.5 #Mesh step
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        plotSVM(linear_svm, 1, "Linear")
        plotSVM(rbf_svm, 2, "RBF")
        plotSVM(sigmoid_svm, 3, "Sigmoid")
        plotSVM(polynomial_svm,  4,  "Polynomial")
        plt.savefig("fotball_trained.png")



# N-Dimensions
if train_nd:
    # Create numpy arrays to use for training. 
    # X contains the features
    # Y contains the classes
    X = np.array(training_0[train_num:] + 
                 training_1[train_num:] + 
                 training_2[train_num:])
    
    Y = np.array([0 for i in training_0][train_num:] + 
                 [1 for i in training_1][train_num:] + 
                 [2 for i in training_2][train_num:])

    print "Starting n-D training..."

    linear_svm = svm.SVC(decision_function_shape = decision_function_shape, kernel = 'linear', C = C, gamma = gamma).fit(X, Y)
    testSVM("Linear", linear_svm,testing_0, testing_1, testing_2)

    rbf_svm = svm.SVC(decision_function_shape = decision_function_shape, kernel = 'rbf', C = C, gamma = gamma).fit(X, Y)
    testSVM("RBF", rbf_svm,testing_0, testing_1, testing_2)

    sigmoid_svm = svm.SVC(decision_function_shape = decision_function_shape, kernel = 'sigmoid', C = C, gamma = gamma).fit(X, Y)
    testSVM("Sigmoid", sigmoid_svm,testing_0, testing_1, testing_2)

    nu_svm = svm.NuSVC().fit(X, Y)
    testSVM("NuSVC", nu_svm,testing_0, testing_1, testing_2)

    polynomial_svm = svm.SVC(decision_function_shape = decision_function_shape, kernel = 'poly', C = C, gamma = gamma, degree = degree).fit(X, Y)
    testSVM("Polynomial", polynomial_svm, testing_0, testing_1, testing_2)

if train_auto:
    # Create numpy arrays to use for training. 
    # X contains the features
    # Y contains the classes
    X = np.array(training_0[train_num:] + 
                 training_1[train_num:] + 
                 training_2[train_num:])
    
    Y = np.array([0 for i in training_0][train_num:] + 
                 [1 for i in training_1][train_num:] + 
                 [2 for i in training_2][train_num:])

    parameters = [{
                    'kernel': ['rbf'], 
                    'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000]
                  },
                  {
                    'kernel': ['linear'], 
                    'C': [1, 10, 100, 1000]
                  },
                  {
                    'kernel': ['sigmoid'], 
                    'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000]
                  },
                  {
                    'kernel': ['poly'], 
                    'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000],
                    'degree': [1]
                  }]

    print "Starting Grid Search training..."

    svr = svm.SVC()
    clf = grid_search.GridSearchCV(svr, parameters)
    _svm = clf.fit(X, Y)
    print _svm.best_params_, "\n"
    testSVM("Grid Search", _svm, testing_0, testing_1, testing_2)