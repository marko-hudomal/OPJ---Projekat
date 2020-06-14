import preprocessing
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn.model_selection import GridSearchCV, KFold, cross_validate

import sys

class Parameters :
    def __init__(self, lowerCaseFlag, removeStopWordsFlag, stemFlag, maxFeatures, ngramRange, tfidfFlags):
        self.lowerCaseFlag = lowerCaseFlag
        self.removeStopWordsFlag = removeStopWordsFlag
        self.stemFlag = stemFlag
        self.maxFeatures = maxFeatures
        self.ngramRange = ngramRange
        self.tfidfFlags = tfidfFlags

def getInfoFromParameters(input_file, parameters):

    Corpus = preprocessing.read_data(input_file, to_lower_case=parameters.lowerCaseFlag, remove_stop_words=parameters.removeStopWordsFlag, stem=parameters.stemFlag)
    matrix, names = preprocessing.vectorize(Corpus, max_features=parameters.maxFeatures, ngram_range=parameters.ngramRange, tf=parameters.tfidfFlags[0], idf=parameters.tfidfFlags[1], tfidf=parameters.tfidfFlags[2])

    #TrainX, TestX, TrainY, TestY = model_selection.train_test_split(matrix, Corpus['Class'],test_size=parameters.testSize)

    return Corpus, matrix, names

def getHeaderAll():

    result = ",Functional-Method Precision,Functional-Method Recall,Functional-Method F1-Score,Functional-Method Support,Functional-Module Precision,Functional-Module Recall,Functional-Module F1-Score,Functional-Module Support," + "Functional-Inline Precision,Functional-Inline Recall,Functional-Inline F1-Score,Functional-Inline Support,Code Precision,Code Recall,Code F1-Score,Code Support,IDE Precision,IDE Recall,IDE F1-Score,IDE Support,"
    result += "General Precision,General Recall,General F1-Score,General Support,Notice Precision,Notice Recall,Notice F1-Score,Notice Support,ToDo Precision,ToDo Recall,ToDo F1-Score,ToDo-Support"

    return result

def getHeaderFunctionalOnly():
    result = ",Functional Precision,Functional Recall,Functional F1-Score,Functional Support,Code Precision,Code Recall,Code F1-Score,Code Support,IDE Precision,IDE Recall,IDE F1-Score,IDE Support,"
    result += "General Precision,General Recall,General F1-Score,General Support,Notice Precision,Notice Recall,Notice F1-Score,Notice Support,ToDo Precision,ToDo Recall,ToDo F1-Score,ToDo-Support"
    
    return result

classificationReportList = []

def scoringFunction(estimator, x, y):

    global classificationReportList

    predictions = estimator.predict(x)
    classificationReport = classification_report(y_pred = predictions, y_true = y, output_dict = True)
    classificationReportList.append([accuracy_score(y_pred = predictions, y_true = y)*100, classificationReport])

    return 1

def printAverageValuesOfClassificationReportList(outputFile, parameters):

    global classificationReportList

    print("Length = ", len(classificationReportList))
    resultDictionary = {}

    commentClassArray = []
    if (outputFile == 'output.csv'):
        commentClassArray = ['Functional-Method', 'Functional-Module', 'Functional-Inline', 'Code', 'IDE', 'General', 'Notice', 'ToDo']
    else:
        commentClassArray = ['Functional', 'Code', 'IDE', 'General', 'Notice', 'ToDo']

    for i in range(0, len(classificationReportList)):
        for commentClass in commentClassArray:
            for factor in ['precision', 'recall', 'f1-score', 'support']:
                if (commentClass in resultDictionary):
                    if (factor in resultDictionary[commentClass]):
                        resultDictionary[commentClass][factor] = resultDictionary[commentClass][factor] + classificationReportList[i][1][commentClass][factor]
                    else:
                        resultDictionary[commentClass][factor] = classificationReportList[i][1][commentClass][factor]
                else:
                    resultDictionary[commentClass] = {}
                    resultDictionary[commentClass][factor] = classificationReportList[i][1][commentClass][factor]  
        
    for commentClass in commentClassArray:
        for factor in ['precision', 'recall', 'f1-score', 'support']:
            resultDictionary[commentClass][factor] = resultDictionary[commentClass][factor] / len(classificationReportList)    
            
    accuracy = 0
    for i in range(0, len(classificationReportList)):
        accuracy = accuracy + classificationReportList[i][0]
    
    accuracy = accuracy / len(classificationReportList)

    if (outputFile == 'output.csv'):
        print(parameters.lowerCaseFlag, parameters.removeStopWordsFlag, parameters.stemFlag, parameters.maxFeatures, parameters.ngramRange, parameters.tfidfFlags[0], parameters.tfidfFlags[1], parameters.tfidfFlags[2], accuracy, 
        resultDictionary['Functional-Method']['precision'], resultDictionary['Functional-Method']['recall'], resultDictionary['Functional-Method']['f1-score'], resultDictionary['Functional-Method']['support'],
        resultDictionary['Functional-Module']['precision'], resultDictionary['Functional-Module']['recall'], resultDictionary['Functional-Module']['f1-score'], resultDictionary['Functional-Module']['support'],
        resultDictionary['Functional-Inline']['precision'], resultDictionary['Functional-Inline']['recall'], resultDictionary['Functional-Inline']['f1-score'], resultDictionary['Functional-Inline']['support'],
        resultDictionary['Code']['precision'], resultDictionary['Code']['recall'], resultDictionary['Code']['f1-score'], resultDictionary['Code']['support'],
        resultDictionary['IDE']['precision'], resultDictionary['IDE']['recall'], resultDictionary['IDE']['f1-score'], resultDictionary['IDE']['support'],
        resultDictionary['General']['precision'], resultDictionary['General']['recall'], resultDictionary['General']['f1-score'], resultDictionary['General']['support'],
        resultDictionary['Notice']['precision'], resultDictionary['Notice']['recall'], resultDictionary['Notice']['f1-score'], resultDictionary['Notice']['support'],
        resultDictionary['ToDo']['precision'], resultDictionary['ToDo']['recall'], resultDictionary['ToDo']['f1-score'], resultDictionary['ToDo']['support'], sep=',')
    else:
        print(parameters.lowerCaseFlag, parameters.removeStopWordsFlag, parameters.stemFlag, parameters.maxFeatures, parameters.ngramRange, parameters.tfidfFlags[0], parameters.tfidfFlags[1], parameters.tfidfFlags[2], accuracy, 
        resultDictionary['Functional']['precision'], resultDictionary['Functional']['recall'], resultDictionary['Functional']['f1-score'], resultDictionary['Functional']['support'],
        resultDictionary['Code']['precision'], resultDictionary['Code']['recall'], resultDictionary['Code']['f1-score'], resultDictionary['Code']['support'],
        resultDictionary['IDE']['precision'], resultDictionary['IDE']['recall'], resultDictionary['IDE']['f1-score'], resultDictionary['IDE']['support'],
        resultDictionary['General']['precision'], resultDictionary['General']['recall'], resultDictionary['General']['f1-score'], resultDictionary['General']['support'],
        resultDictionary['Notice']['precision'], resultDictionary['Notice']['recall'], resultDictionary['Notice']['f1-score'], resultDictionary['Notice']['support'],
        resultDictionary['ToDo']['precision'], resultDictionary['ToDo']['recall'], resultDictionary['ToDo']['f1-score'], resultDictionary['ToDo']['support'], sep=',')

    classificationReportList = []

if __name__ == "__main__":

    # Construct parameters.
    parametersList = list()

    for lowerCaseFlag in [True]:
        for removeStopWordsFlag in [False, True]:
            for stemFlag in [False, True]:
                    for maxFeatures in [1000, 5000]:
                        for ngramRange in [(1, 1), (1, 2), (1, 3)]:
                            for tfidfFlags in [(False, False, False), (True, False, False), (False, False, True)]:
                                parametersList.append(Parameters(
                                    lowerCaseFlag, 
                                    removeStopWordsFlag, 
                                    stemFlag, 
                                    maxFeatures,
                                    ngramRange,
                                    tfidfFlags)
                                )

    original_stdout = sys.stdout # Save a reference to the original standard output

    # Go through all of the input files and configurations and export the results to a .csv file.
    # ("input-functional.txt", "output-functional.csv")
    for input_file, output_file in [("input.txt", "output.csv")]:
         with open(output_file, 'w') as output:
            sys.stdout = output
            
            header = "Lower Case,Remove Stop Words,Stem,Test Size,Max Features,N-gram Range,TF,IDF,TFIDF,Accuracy"
            if (input_file == "input.txt"):
                header += getHeaderAll()
            else:
                header += getHeaderFunctionalOnly()
            print(header)

            for parameters in parametersList:
                # Find optimal hyperparameter C and gamma.
                Corpus, matrix, names = getInfoFromParameters("input-functional.txt", parameters)

                # [0.1, 1, 10, 100]
                param_grid = {'C': [0.1, 1, 10],
                            'kernel': ['linear']}

                inner_cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
                outer_cv = KFold(n_splits = 10, shuffle = True, random_state = 42)

                # Inner CV.
                SVM = GridSearchCV(svm.SVC(), param_grid, refit = True, cv=inner_cv, verbose = 0)

                # Outer CV. SVM.fit() gets called in cross_validate.
                cross_validate(SVM, X=matrix, y=Corpus['Class'], scoring = scoringFunction, cv = outer_cv, return_train_score = True)

                printAverageValuesOfClassificationReportList("output-functional.csv", parameters)

            sys.stdout = original_stdout
