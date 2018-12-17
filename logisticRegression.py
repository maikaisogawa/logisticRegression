
# File: Logistic Regression - Gradient Ascent Algorithm
# ----------------------
# This is code for CS109 problem set 6
# Maika Isogawa

import numpy as np
import math
from tqdm import tqdm

def main():
    
    # change these based on question parameters
    steps = 10000
    learningRate = 0.000001
    
    ### training portion ###
    with open("heart-train.txt") as trainingData: #change file name depending on data
        lines = trainingData.readlines()
    numInputVariables = int(lines[0])
    numDataVectors = int(lines[1])
    xVector, yVector = makeVectors(lines)

    # initialize thetas to 0
    thetas = [0] * (numInputVariables + 1)

# print(logLikelihood(thetas, xVector, yVector, numDataVectors))

    # Repeat many times
    for i in tqdm(range(steps)):
        #initialize gradient to 0
        gradient = [0] * (numInputVariables + 1)
        # calculate graidents for each training example
        for j in range(numDataVectors):
            cur = yVector[j] - sig(thetaTransposeX(thetas, xVector[j]))
            for k in range(numInputVariables + 1):
                gradient[k] += cur * xVector[j][k]
        for j in range(len(thetas)):
            thetas[j] += learningRate * gradient[j]
    print(thetas)


    ### testing portion ###
    with open("heart-test.txt") as testingData: #change file name depending on data
        testingLines = testingData.readlines()
    numInputVariables = int(testingLines[0])
    numDataVectors = int(testingLines[1])
    testY0, testY1 = makeYClasses(testingLines, numInputVariables, numDataVectors)
    y1Correct = 0
    y0Correct = 0
    for vector in testY1:
        if (sig(thetaTransposeX(thetas, vector)) >= 0.5):
            y1Correct += 1
    for vector in testY0:
        if (sig(thetaTransposeX(thetas, vector)) < 0.5):
            y0Correct += 1

    # print results
    print("Class 0: tested ", str(len(testY0)), ", correctly classified ", str(y0Correct))
    print("Class 1: tested ", str(len(testY1)), ", correctly classified ", str(y1Correct))
    print("Overall: tested ", str(len(testY0) + len(testY1)), ", correctly classified ", str(y0Correct + y1Correct))
    print("Accuracy = ", str(float(y0Correct+y1Correct)/(len(testY0) + len(testY1))))

#print("log likelihoods: ", logLikelihood(thetas, xVector, yVector, numDataVectors))



# reads and parses data, returns the x and y vectors
def makeVectors(lines):
    xVector = []
    yVector = []
    numVectors = 0
    for i in range(len(lines)):
        if i > 1:
            # each line of data of x v y is split by ':' , and x is split by ' '
            xData = lines[i].split(': ')[0]   # the portion before ': ' is the x values
            yClass = lines[i].split(': ')[1]    # the portion after ': ' is the y class
            xVector.append(xData.split(' '))
            yVector.append(yClass)
    for i in range(len(xVector)):
        for j in range(len(xVector[i])):
            xVector[i][j] = float(xVector[i][j]) # make into numbers
        xVector[i] = [1] + xVector[i]
        for i in range(len(yVector)):
            yVector[i] = float(yVector[i]) #make into numbers
    return xVector, yVector

# returns arrays of binary classes for Y
def makeYClasses(lines, numInputVariables, numDataVectors):
    testY0 = []
    testY1 = []
    for i in range(len(lines)):
        if i > 1:
            vector = lines[i].split(': ')[0]   # the portion before ': ' is the vector data values
            yClass = lines[i].split(': ')[1]    # the portion after ': ' is the y class
            if yClass[0] == '1':
                testY1.append(vector.split(' '))
            elif yClass[0] == '0':
                testY0.append(vector.split(' '))
    for i in range(len(testY0)):
        for j in range(len(testY0[i])):
            testY0[i][j] = float(testY0[i][j]) # make into numbers
        testY0[i] = [1] + testY0[i]
    for i in range(len(testY1)):
        for j in range(len(testY1[i])):
            testY1[i][j] = float(testY1[i][j]) # make into numbers
            testY1[i] = [1] + testY1[i]
    return testY0, testY1

# calculate the log likelihood
def logLikelihood(thetas, xVector, yVector, numDataVectors):
    LL = 0
    for i in range(numDataVectors):
        yHat = sig(thetaTransposeX(thetas, xVector[i]))
        LL += (yVector[i]*math.log(yHat)) + ((1-yVector[i])*math.log(1-yHat))
    return LL

# returns the result of the sigmoid function
def sig(x):
    return (1 / float(1 + math.exp(-x)))
                               
# returns theta transpose x
def thetaTransposeX(thetas, vector):
    result = 0
    for i in range(len(thetas)):
        result += thetas[i] * vector[i]
    return result

if __name__ == '__main__':
    main()
