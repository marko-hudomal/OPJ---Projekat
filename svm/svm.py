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
from sklearn.model_selection import GridSearchCV 

import sys

class Parameters :
    def __init__(self, lowerCaseFlag, removeStopWordsFlag, stemFlag, testSize, maxFeatures, ngramRange, tfidfFlags):
        self.lowerCaseFlag = lowerCaseFlag
        self.removeStopWordsFlag = removeStopWordsFlag
        self.stemFlag = stemFlag
        self.testSize = testSize
        self.maxFeatures = maxFeatures
        self.ngramRange = ngramRange
        self.tfidfFlags = tfidfFlags

def getInfoFromParameters(input_file, parameters):

    Corpus = preprocessing.read_data(input_file, to_lower_case=parameters.lowerCaseFlag, remove_stop_words=parameters.removeStopWordsFlag, stem=parameters.stemFlag)
    matrix, names = preprocessing.vectorize(Corpus, max_features=parameters.maxFeatures, ngram_range=parameters.ngramRange, tf=parameters.tfidfFlags[0], idf=parameters.tfidfFlags[1], tfidf=parameters.tfidfFlags[2])
    TrainX, TestX, TrainY, TestY = model_selection.train_test_split(matrix, Corpus['Class'],test_size=parameters.testSize)

    return Corpus, matrix, names, TrainX, TestX, TrainY, TestY

def getHeaderAll():

    result = ",Functional-Method Precision,Functional-Method Recall,Functional-Method F1-Score,Functional-Method Support,Functional-Module Precision,Functional-Module Recall,Functional-Module F1-Score,Functional-Module Support," + "Functional-Inline Precision,Functional-Inline Recall,Functional-Inline F1-Score,Functional-Inline Support,Code Precision,Code Recall,Code F1-Score,Code Support,IDE Precision,IDE Recall,IDE F1-Score,IDE Support,"
    result += "General Precision,General Recall,General F1-Score,General Support,Notice Precision,Notice Recall,Notice F1-Score,Notice Support"

    return result

def getHeaderFunctionalOnly():
    result = ",Functional Precision,Functional Recall,Functional F1-Score,Functional Support,Code Precision,Code Recall,Code F1-Score,Code Support,IDE Precision,IDE Recall,IDE F1-Score,IDE Support,"
    result += "General Precision,General Recall,General F1-Score,General Support,Notice Precision,Notice Recall,Notice F1-Score,Notice Support"
    
    return result

if __name__ == "__main__":

    # Construct parameters.
    parametersList = list()

    for lowerCaseFlag in [True]:
        for removeStopWordsFlag in [False, True]:
            for stemFlag in [False]:
                for testSize in [0.25, 0.3, 0.35]:
                    for maxFeatures in [1000, 5000]:
                        for ngramRange in [(1, 1), (1, 2), (1, 3)]:
                            for tfidfFlags in [(False, False, False), (True, False, False), (False, False, True)]:
                                parametersList.append(Parameters(
                                    lowerCaseFlag, 
                                    removeStopWordsFlag, 
                                    stemFlag, 
                                    testSize, 
                                    maxFeatures,
                                    ngramRange,
                                    tfidfFlags)
                                )

    original_stdout = sys.stdout # Save a reference to the original standard output

    # Find optimal hyperparameter C and gamma.
    Corpus, matrix, names, TrainX, TestX, TrainY, TestY = getInfoFromParameters("input.txt", parametersList[0])
    svc = svm.SVC()
    svc.fit(TrainX, TrainY)

    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
                'gamma': [1, 0.1, 0.01], 
                'kernel': ['linear']}

    SVM = GridSearchCV(svc, param_grid, refit = True, verbose = 3, n_jobs=6)
    SVM.fit(TrainX, TrainY)

    optimalC = SVM.best_estimator_.C
    optimalGamma = SVM.best_estimator_.gamma

    # Go through all of the input files and configurations and export the results to a .csv file.
    for input_file, output_file in [("input.txt", "output.csv"), ("input-functional.txt", "output-functional.csv")]:
         with open(output_file, 'w') as output:
            sys.stdout = output
            
            header = "Lower Case,Remove Stop Words,Stem,Test Size,Max Features,N-gram Range,TF,IDF,TFIDF,Accuracy"
            if (input_file == "input.txt"):
                header += getHeaderAll()
            else:
                header += getHeaderFunctionalOnly()
            print(header)

            for parameters in parametersList:
                
                Corpus, matrix, names, TrainX, TestX, TrainY, TestY = getInfoFromParameters(input_file, parametersList[0])
                svc = svm.SVC(C=optimalC, kernel='linear', gamma=optimalGamma, decision_function_shape='ovr')

                svc.fit(TrainX, TrainY)

                predictions_SVM = svc.predict(TestX)

                classificationReport = classification_report(y_pred = predictions_SVM, y_true = TestY, output_dict = True)

                if (input_file == "input.txt"):
                    print(parameters.lowerCaseFlag, parameters.removeStopWordsFlag, parameters.stemFlag, parameters.testSize,
                    parameters.maxFeatures, parameters.ngramRange, parameters.tfidfFlags[0], parameters.tfidfFlags[1], 
                    parameters.tfidfFlags[2], accuracy_score(y_pred = predictions_SVM, y_true = TestY)*100, 
                    classificationReport['Functional-Method']['precision'], classificationReport['Functional-Method']['recall'],classificationReport['Functional-Method']['f1-score'], classificationReport['Functional-Method']['support'],
                    classificationReport['Functional-Module']['precision'], classificationReport['Functional-Module']['recall'],classificationReport['Functional-Module']['f1-score'], classificationReport['Functional-Module']['support'],
                    classificationReport['Functional-Inline']['precision'], classificationReport['Functional-Inline']['recall'],classificationReport['Functional-Inline']['f1-score'], classificationReport['Functional-Inline']['support'],
                    classificationReport['Code']['precision'], classificationReport['Code']['recall'],classificationReport['Code']['f1-score'], classificationReport['Code']['support'],
                    classificationReport['IDE']['precision'], classificationReport['IDE']['recall'],classificationReport['IDE']['f1-score'], classificationReport['IDE']['support'],
                    classificationReport['General']['precision'], classificationReport['General']['recall'],classificationReport['General']['f1-score'], classificationReport['General']['support'],
                    classificationReport['Notice']['precision'], classificationReport['Notice']['recall'],classificationReport['Notice']['f1-score'], classificationReport['Notice']['support'],
                     sep=',')
                else:
                    print(parameters.lowerCaseFlag, parameters.removeStopWordsFlag, parameters.stemFlag, parameters.testSize,
                    parameters.maxFeatures, parameters.ngramRange, parameters.tfidfFlags[0], parameters.tfidfFlags[1], 
                    parameters.tfidfFlags[2], accuracy_score(y_pred = predictions_SVM, y_true = TestY)*100, 
                    classificationReport['Functional']['precision'], classificationReport['Functional']['recall'],classificationReport['Functional']['f1-score'], classificationReport['Functional']['support'],
                    classificationReport['Code']['precision'], classificationReport['Code']['recall'],classificationReport['Code']['f1-score'], classificationReport['Code']['support'],
                    classificationReport['IDE']['precision'], classificationReport['IDE']['recall'],classificationReport['IDE']['f1-score'], classificationReport['IDE']['support'],
                    classificationReport['General']['precision'], classificationReport['General']['recall'],classificationReport['General']['f1-score'], classificationReport['General']['support'],
                    classificationReport['Notice']['precision'], classificationReport['Notice']['recall'],classificationReport['Notice']['f1-score'], classificationReport['Notice']['support'],
                     sep=',')

            sys.stdout = original_stdout
