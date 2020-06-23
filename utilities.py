import preprocessing
import sys
import pandas as pd
import numpy as np
import time
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
from sklearn.naive_bayes import BernoulliNB

class Parameters :
    def __init__(self, lowerCaseFlag, removeStopWordsFlag, stemFlag, maxFeatures, ngramRange, tfidfFlags, alphaNaiveBayes=0.001):
        self.lowerCaseFlag = lowerCaseFlag
        self.removeStopWordsFlag = removeStopWordsFlag
        self.stemFlag = stemFlag
        self.maxFeatures = maxFeatures
        self.ngramRange = ngramRange
        self.tfidfFlags = tfidfFlags
        self.alphaNaiveBayes = alphaNaiveBayes

def getInfoFromParameters(input_file, parameters):
    Corpus = preprocessing.process_data(input_file, to_lower_case=parameters.lowerCaseFlag, remove_stop_words=parameters.removeStopWordsFlag, stem=parameters.stemFlag)
    counts_by_comment, names = preprocessing.vectorize(Corpus, max_features=parameters.maxFeatures, ngram_range=parameters.ngramRange, tf=parameters.tfidfFlags[0], tfidf=parameters.tfidfFlags[1])

    return Corpus, counts_by_comment, names

classificationReportList = []
def scoringFunction(estimator, x, y):
    global classificationReportList

    # estimator.predict(x) calls estimator.best_estimator_.predict(x)
    predictions = estimator.predict(x)
    classificationReport = classification_report(y_pred = predictions, y_true = y, output_dict = True)
    classificationReportList.append(classificationReport)

    return 1

def getHeader(is_functional):
    result = "Lower Case,Remove Stop Words,Stem,Max Features,N-gram Range,TF,TFIDF,Alpha_Naive_Bayes,Accuracy,Macro Precision,Macro Recall,Macro F1-Score"
    if (is_functional):
        result += ",Functional Precision,Functional Recall,Functional F1-Score,Functional Support"
    else:
        result += ",Functional-Method Precision,Functional-Method Recall,Functional-Method F1-Score,Functional-Method Support" + ",Functional-Module Precision,Functional-Module Recall,Functional-Module F1-Score,Functional-Module Support" + ",Functional-Inline Precision,Functional-Inline Recall,Functional-Inline F1-Score,Functional-Inline Support"

    result += ",Code Precision,Code Recall,Code F1-Score,Code Support" + ",IDE Precision,IDE Recall,IDE F1-Score,IDE Support" + ",General Precision,General Recall,General F1-Score,General Support" + ",Notice Precision,Notice Recall,Notice F1-Score,Notice Support" + ",ToDo Precision,ToDo Recall,ToDo F1-Score,ToDo Support"
    return result

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def printAverageValuesOfClassificationReportList(outputFile, parameters, functionalOnlyFlag):

    global classificationReportList
    resultDictionary = {}

    commentClassArray = []
    if (functionalOnlyFlag == False):
        commentClassArray = ['Functional-Method', 'Functional-Module', 'Functional-Inline', 'Code', 'IDE', 'General', 'Notice', 'ToDo']
    else:
        commentClassArray = ['Functional', 'Code', 'IDE', 'General', 'Notice', 'ToDo']

    for i in range(0, len(classificationReportList)):
        for commentClass in commentClassArray:
            for factor in ['precision', 'recall', 'f1-score', 'support']:
                if (commentClass in resultDictionary):
                    if (factor in resultDictionary[commentClass]):
                        resultDictionary[commentClass][factor] = resultDictionary[commentClass][factor] + classificationReportList[i][commentClass][factor]
                    else:
                        resultDictionary[commentClass][factor] = classificationReportList[i][commentClass][factor]
                else:
                    resultDictionary[commentClass] = {}
                    resultDictionary[commentClass][factor] = classificationReportList[i][commentClass][factor]

    for commentClass in commentClassArray:
        for factor in ['precision', 'recall', 'f1-score', 'support']:
            resultDictionary[commentClass][factor] = resultDictionary[commentClass][factor] / len(classificationReportList)
    

    accuracy = 0
    precisionAvg = 0
    recallAvg = 0
    f1ScoreAvg = 0
    for i in range(0, len(classificationReportList)):
        accuracy += classificationReportList[i]['accuracy']
        precisionAvg += classificationReportList[i]['macro avg']['precision']
        recallAvg += classificationReportList[i]['macro avg']['recall']
        f1ScoreAvg += classificationReportList[i]['macro avg']['f1-score']

    accuracy /= len(classificationReportList)
    accuracy *= 100 # (%)
    precisionAvg /= len(classificationReportList)
    recallAvg /= len(classificationReportList)
    f1ScoreAvg /= len(classificationReportList)

    if (functionalOnlyFlag == False):
        print(parameters.lowerCaseFlag, parameters.removeStopWordsFlag, parameters.stemFlag, parameters.maxFeatures, parameters.ngramRange, parameters.tfidfFlags[0], parameters.tfidfFlags[1], parameters.alphaNaiveBayes, accuracy, precisionAvg, recallAvg, f1ScoreAvg,
        resultDictionary['Functional-Method']['precision'], resultDictionary['Functional-Method']['recall'], resultDictionary['Functional-Method']['f1-score'], resultDictionary['Functional-Method']['support'],
        resultDictionary['Functional-Module']['precision'], resultDictionary['Functional-Module']['recall'], resultDictionary['Functional-Module']['f1-score'], resultDictionary['Functional-Module']['support'],
        resultDictionary['Functional-Inline']['precision'], resultDictionary['Functional-Inline']['recall'], resultDictionary['Functional-Inline']['f1-score'], resultDictionary['Functional-Inline']['support'],
        resultDictionary['Code']['precision'], resultDictionary['Code']['recall'], resultDictionary['Code']['f1-score'], resultDictionary['Code']['support'],
        resultDictionary['IDE']['precision'], resultDictionary['IDE']['recall'], resultDictionary['IDE']['f1-score'], resultDictionary['IDE']['support'],
        resultDictionary['General']['precision'], resultDictionary['General']['recall'], resultDictionary['General']['f1-score'], resultDictionary['General']['support'],
        resultDictionary['Notice']['precision'], resultDictionary['Notice']['recall'], resultDictionary['Notice']['f1-score'], resultDictionary['Notice']['support'],
        resultDictionary['ToDo']['precision'], resultDictionary['ToDo']['recall'], resultDictionary['ToDo']['f1-score'], resultDictionary['ToDo']['support'], file=outputFile, sep=',')
    else:
        print(parameters.lowerCaseFlag, parameters.removeStopWordsFlag, parameters.stemFlag, parameters.maxFeatures, parameters.ngramRange, parameters.tfidfFlags[0], parameters.tfidfFlags[1], parameters.alphaNaiveBayes, accuracy, precisionAvg, recallAvg, f1ScoreAvg,
        resultDictionary['Functional']['precision'], resultDictionary['Functional']['recall'], resultDictionary['Functional']['f1-score'], resultDictionary['Functional']['support'],
        resultDictionary['Code']['precision'], resultDictionary['Code']['recall'], resultDictionary['Code']['f1-score'], resultDictionary['Code']['support'],
        resultDictionary['IDE']['precision'], resultDictionary['IDE']['recall'], resultDictionary['IDE']['f1-score'], resultDictionary['IDE']['support'],
        resultDictionary['General']['precision'], resultDictionary['General']['recall'], resultDictionary['General']['f1-score'], resultDictionary['General']['support'],
        resultDictionary['Notice']['precision'], resultDictionary['Notice']['recall'], resultDictionary['Notice']['f1-score'], resultDictionary['Notice']['support'],
        resultDictionary['ToDo']['precision'], resultDictionary['ToDo']['recall'], resultDictionary['ToDo']['f1-score'], resultDictionary['ToDo']['support'], file=outputFile, sep=',')

    classificationReportList = []

