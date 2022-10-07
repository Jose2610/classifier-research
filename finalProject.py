# Programmer: Jose Luis Sanchez
# Date Modified: 6/7/2022
# Purpose: Project 4, Individual Project

# Quick note on this project, I ended up having quite a lot of 'ConvergenceWarning' due
# to the classifier not converging. Though this doesn't really lower the accuracy
# too much, I due believe it's due to the data being binary and having an issue with
# that.

# Imports
import time
import random
import copy
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import numpy

# Import neural network stuff
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

# Global Variables
MATRIX_SIZE = 49
# N_SAMPLES will have three different testing values, 10/20/30
N_SAMPLES = 30
# NOISE_RATE is from 0 percent to 100 percent
# NOISE_RATE will also have three different testing values, 20/50/85
NOISE_RATE = 85
TOTAL_SAMPLES = N_SAMPLES**2
TOTAL_TESTS = 20

# generateNoise function takes a character prototype and generates noise
# to randomize locations of the matrix and inverts the bit value
def generateNoise(matrix, NOISE_RATE, ider):
    sampleSet = []
    index = 0
    noise = int(round((NOISE_RATE * MATRIX_SIZE) / 100.0))
    sample = copy.deepcopy(matrix)
    while index < N_SAMPLES:
        nonRepeat = []
        sampleSet.append(sample)
        sample = copy.deepcopy(matrix)
        temp = random.sample(list(enumerate(sample)), noise)
        for idx, val in temp:
            nonRepeat.append(idx)
        for j in range(noise):
            sample[nonRepeat[j]] = 0 if nonRepeat[j] == 1 else 1
        index += 1
    return sampleSet

# printMatrix function will print the matrix representation of the 
# character
def printMatrix(matrix):
    for i in range(MATRIX_SIZE):
        if i % 7 == 0:
            print()
        print(matrix[i], end='')

# Classifier functions

# mlpClassifier function (Current problem with ConvergenceWarning, but still finds results)
def mlpClassifier(xTrain, xTest, xLabel, actualLabel):
    # Start MLP with gridsearch
    averageTime = []
    testAccuracy = []
    paramMLP = {
        'hidden_layer_sizes': [(25), (25, 15), (25, 15, 5)],
        'max_iter': [50, 100, 150],
        'activation': ['tanh'],
        'solver': ['sgd'],
        'learning_rate': ['constant'],
        'learning_rate_init': [0.1, 0.3, 0.5]
    }
    mlpClf = MLPClassifier()
    gridMLP = GridSearchCV(mlpClf, paramMLP, n_jobs=-1)
    gridMLP.fit(xTrain, xLabel)
    newClf = MLPClassifier(**gridMLP.best_params_)
    for i in range(TOTAL_TESTS):
        start = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            newClf.fit(xTrain, xLabel)
        stop = time.time()
        totalTime = stop - start
        averageTime.append(totalTime)
        yPred = newClf.predict(xTest)
        testAccuracy.append(accuracy_score(actualLabel, yPred) * 100)
    print(gridMLP.best_params_)

    totalAverageTime = sum(averageTime) / len(averageTime)
    totalTestAccuracy = sum(testAccuracy) / len(testAccuracy)
    print('Training time: {:.5f}s'.format(totalAverageTime))
    print('Accuracy: {:.2f}%'.format(totalTestAccuracy))

# knnClassifier function
def knnClassifier(xTrain, xTest, xLabel, actualLabel):
    # Start KNN with gridsearch
    averageTime = []
    testAccuracy = []
    kRange = list(range(1, 31))
    paramKNN = {
        'n_neighbors': kRange,
        'metric': ['euclidean', 'manhattan', 'minkowski'], 
    }
    knnClf = KNeighborsClassifier()
    gridKNN = GridSearchCV(knnClf, paramKNN, n_jobs=-1)
    gridKNN.fit(xTrain, xLabel)
    newClf = KNeighborsClassifier(**gridKNN.best_params_)
    for i in range(TOTAL_TESTS):
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            newClf.fit(xTrain, xLabel)
        stop = time.time()
        totalTime = stop - start
        averageTime.append(totalTime)
        yPred = newClf.predict(xTest)
        testAccuracy.append(accuracy_score(actualLabel, yPred) * 100)
    print(gridKNN.best_params_)

    totalAverageTime = sum(averageTime) / len(averageTime)
    totalTestAccuracy = sum(testAccuracy) / len(testAccuracy)
    print('Training time: {:.5f}s'.format(totalAverageTime))
    print('Accuracy: {:.2f}%'.format(totalTestAccuracy))

# svmClassifier function (using SVC function)
def svmClassifier(xTrain, xTest, xLabel, actualLabel):
    # Start SVM with gridsearch
    averageTime = []
    testAccuracy = []
    paramSVM = {
        'C': [ 1, 10, 100, 1000], 
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }
    svmClf = svm.SVC()
    gridSVM = GridSearchCV(svmClf, paramSVM, n_jobs=-1)
    gridSVM.fit(xTrain, xLabel)
    newClf = svm.SVC(**gridSVM.best_params_)
    for i in range(TOTAL_TESTS):
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            newClf.fit(xTrain, xLabel)
        stop = time.time()
        totalTime = stop - start
        averageTime.append(totalTime)
        yPred = newClf.predict(xTest)
        testAccuracy.append(accuracy_score(actualLabel, yPred) * 100)
    print(gridSVM.best_params_)

    totalAverageTime = sum(averageTime) / len(averageTime)
    totalTestAccuracy = sum(testAccuracy) / len(testAccuracy)
    print('Training time: {:.5f}s'.format(totalAverageTime))
    print('Accuracy: {:.2f}%'.format(totalTestAccuracy))

# rfClassifier
def rfClassifier(xTrain, xTest, xLabel, actualLabel):
    # Start SVM with gridsearch
    averageTime = []
    testAccuracy = []
    paramRF = {'n_estimators': [25, 50, 75, 100, 200],
               'max_depth': [10, 25, None],
               'min_samples_split': [2, 3],
               'min_samples_leaf': [1, 2],
    }
    rfClf = RandomForestClassifier()
    gridRF = GridSearchCV(rfClf, paramRF, n_jobs=-1)
    gridRF.fit(xTrain, xLabel)
    newClf = RandomForestClassifier(**gridRF.best_params_)
    for i in range(TOTAL_TESTS):
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            newClf.fit(xTrain, xLabel)
        stop = time.time()
        totalTime = stop - start
        averageTime.append(totalTime)
        yPred = newClf.predict(xTest)
        testAccuracy.append(accuracy_score(actualLabel, yPred) * 100)
    print(gridRF.best_params_)

    totalAverageTime = sum(averageTime) / len(averageTime)
    totalTestAccuracy = sum(testAccuracy) / len(testAccuracy)
    print('Training time: {:.5f}s'.format(totalAverageTime))
    print('Accuracy: {:.2f}%'.format(totalTestAccuracy))

# Character prototypes
protoA = [0,0,0,1,0,0,0, 0,0,1,0,1,0,0, 0,0,1,0,1,0,0, 0,1,0,0,0,1,0, 0,1,1,1,1,1,0, 1,0,0,0,0,0,1, 1,0,0,0,0,0,1]
protoB = [1,1,1,1,1,1,0, 1,0,0,0,0,0,1, 1,0,0,0,0,0,1, 1,1,1,1,1,1,0, 1,0,0,0,0,0,1, 1,0,0,0,0,0,1, 1,1,1,1,1,1,0]
protoC = [0,1,1,1,1,1,0, 1,0,0,0,0,0,1, 1,0,0,0,0,0,0, 1,0,0,0,0,0,0, 1,0,0,0,0,0,0, 1,0,0,0,0,0,1, 0,1,1,1,1,1,0]
protoD = [1,1,1,1,1,1,0, 1,0,0,0,0,0,1, 1,0,0,0,0,0,1, 1,0,0,0,0,0,1, 1,0,0,0,0,0,1, 1,0,0,0,0,0,1, 1,1,1,1,1,1,0]
protoE = [1,1,1,1,1,1,1, 1,0,0,0,0,0,0, 1,0,0,0,0,0,0, 1,1,1,1,1,1,0, 1,0,0,0,0,0,0, 1,0,0,0,0,0,0, 1,1,1,1,1,1,1]
protoF = [1,1,1,1,1,1,1, 1,0,0,0,0,0,0, 1,0,0,0,0,0,0, 1,1,1,1,1,1,0, 1,0,0,0,0,0,0, 1,0,0,0,0,0,0, 1,0,0,0,0,0,0]
protoG = [0,1,1,1,1,1,0, 1,0,0,0,0,0,1, 1,0,0,0,0,0,0, 1,0,0,1,1,1,1, 1,0,0,0,0,0,1, 1,0,0,0,0,0,1, 0,1,1,1,1,1,0]
protoH = [1,0,0,0,0,0,1, 1,0,0,0,0,0,1, 1,0,0,0,0,0,1, 1,1,1,1,1,1,1, 1,0,0,0,0,0,1, 1,0,0,0,0,0,1, 1,0,0,0,0,0,1]
protoI = [1,1,1,1,1,1,1, 0,0,0,1,0,0,0, 0,0,0,1,0,0,0, 0,0,0,1,0,0,0, 0,0,0,1,0,0,0, 0,0,0,1,0,0,0, 1,1,1,1,1,1,1]
protoJ = [0,0,0,0,0,0,1, 0,0,0,0,0,0,1, 0,0,0,0,0,0,1, 0,0,0,0,0,0,1, 0,0,0,0,0,0,1, 1,0,0,0,0,0,1, 0,1,1,1,1,1,0]

# Generate noisy variants
aSet = generateNoise(protoA, NOISE_RATE, 'a')
bSet = generateNoise(protoB, NOISE_RATE, 'b')
cSet = generateNoise(protoC, NOISE_RATE, 'c')
dSet = generateNoise(protoD, NOISE_RATE, 'd')
eSet = generateNoise(protoE, NOISE_RATE, 'e')
fSet = generateNoise(protoF, NOISE_RATE, 'f')
gSet = generateNoise(protoG, NOISE_RATE, 'g')
hSet = generateNoise(protoH, NOISE_RATE, 'h')
iSet = generateNoise(protoI, NOISE_RATE, 'i')
jSet = generateNoise(protoJ, NOISE_RATE, 'j')
mainSet = aSet + bSet + cSet + dSet + eSet + fSet + gSet + hSet + iSet + jSet

# Save mainSet dataset into a csv file for classification purposes
data = pd.read_csv("Datasets\set30noise50.csv").to_numpy()

# Save data into respective variables
# 60 for 100 sample, 120 for 200 sample, and 180 for 300 sample
whichOne = [60, 120, 180]
xTrain = data[0:whichOne[2], 1:]
xLabel = data[0:whichOne[2], 0]
xTest = data[whichOne[2]:, 1:]
actualLabel = data[whichOne[2]:, 0]

# Feature scaling for preprocessing
sc = StandardScaler()
scaler = sc.fit(xTrain)
xTrain = scaler.transform(xTrain)
xTest = scaler.transform(xTest)

# Start testing
# Comment tests after getting results
# MLP
print()
print('MLP Classifier')
#mlpClassifier(xTrain, xTest, xLabel, actualLabel)

# KNN
print()
print('KNN Classifier')
#knnClassifier(xTrain, xTest, xLabel, actualLabel)

# SVM
print()
print('SVM Classifier')
#svmClassifier(xTrain, xTest, xLabel, actualLabel)

# Random Forest
print()
print('Random Forest Classifier')
#rfClassifier(xTrain, xTest, xLabel, actualLabel)

# Wrapper Feature Selection Method
# Using Sequential Forward Selection
print('Running wrapper feature selection...')
params = {
    'C': [ 1, 10, 100, 1000], 
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}
feature_names = ('bit1', 'bit2', 'bit3', 'bit4', 'bit5', 'bit6', 'bit7', 'bit8', 'bit9', 'bit10', 'bit11', 'bit12', 'bit13', 'bit14', 'bit15', 'bit16', 'bit17', 'bit18', 'bit19', 'bit20', 'bit21', 'bit22', 'bit23', 'bit24', 'bit25', 'bit26', 'bit27', 'bit28', 'bit29', 'bit30', 'bit31', 'bit32', 'bit33', 'bit34', 'bit35', 'bit36', 'bit37', 'bit38', 'bit39', 'bit40', 'bit41', 'bit42', 'bit43', 'bit44', 'bit45', 'bit46', 'bit47', 'bit48', 'bit49')
clf = svm.SVC()
gridSVM = GridSearchCV(clf, params, n_jobs=-1)
gridSVM.fit(xTrain, xLabel)
newClf = svm.SVC(**gridSVM.best_params_)

sfs1 = SFS(estimator=newClf, k_features=(1, 49), forward=True, floating=False, scoring='accuracy', cv=5)
sfs1.fit(xTrain, xLabel, custom_feature_names=feature_names)

print('Best combination for sample size %i and noise rate %i (Accuracy: %.3f): %s\n' % (300, 85, sfs1.k_score_, sfs1.k_feature_names_))

fig = plot_sfs(sfs1.get_metric_dict(), kind='std_dev')

plt.ylim([0, 1])
plt.title('Wrapper Feature Selection for SVM (Samples 300/Noise 85)')
plt.grid()
plt.show()