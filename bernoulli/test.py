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
from sklearn.naive_bayes import BernoulliNB

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
    Corpus = preprocessing.process_data(input_file, to_lower_case=parameters.lowerCaseFlag, remove_stop_words=parameters.removeStopWordsFlag, stem=parameters.stemFlag)
    counts_by_comment, names = preprocessing.vectorize(Corpus, max_features=parameters.maxFeatures, ngram_range=parameters.ngramRange, tf=parameters.tfidfFlags[0], tfidf=parameters.tfidfFlags[1])

    return Corpus, counts_by_comment, names

def getHeaderAll():
    result = ",Functional-Method Precision,Functional-Method Recall,Functional-Method F1-Score,Functional-Method Support,Functional-Module Precision,Functional-Module Recall,Functional-Module F1-Score,Functional-Module Support," + "Functional-Inline Precision,Functional-Inline Recall,Functional-Inline F1-Score,Functional-Inline Support,Code Precision,Code Recall,Code F1-Score,Code Support,IDE Precision,IDE Recall,IDE F1-Score,IDE Support,"
    result += "General Precision,General Recall,General F1-Score,General Support,Notice Precision,Notice Recall,Notice F1-Score,Notice Support"

    return result

def getHeaderFunctionalOnly():
    result = ",Functional Precision,Functional Recall,Functional F1-Score,Functional Support,Code Precision,Code Recall,Code F1-Score,Code Support,IDE Precision,IDE Recall,IDE F1-Score,IDE Support,"
    result += "General Precision,General Recall,General F1-Score,General Support,Notice Precision,Notice Recall,Notice F1-Score,Notice Support"

    return result

if __name__ == "__main__":

    print("----------------------------------- TEST START -----------------------------------")

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

    print("Using test param[0]")

    # Find optimal params.
    fileData = preprocessing.read_file("../input-functional.txt")
    Corpus, X, names = getInfoFromParameters(fileData, parametersList[0])
    # Encoder = LabelEncoder()
    # Y = Encoder.fit_transform(Corpus["Class"])
    Y = Corpus["Class"]
    TrainX, TestX, TrainY, TestY = model_selection.train_test_split(X, Y,test_size=parametersList[0].testSize)

    # Some prints.
    # print("Corpus[Comment][1]: ",Corpus["Comment"][1])
    # print("Corpus[Class][1]: ",Corpus["Class"][1])
    # print("X[1]: ",X[1])
    # print("Y[1]: ",Y[1])
    # print("names: ",names)

    # print("Corpus[Class]: ",Corpus["Class"])
    # print("Y: ",Y)

    bernoulli = BernoulliNB()
    bernoulli.fit(TrainX,TrainY)

    print("Search for best estimator params...")
    # [1] Find optimal params
    # param_grid = {'alpha': [0.0001,0.001, 0.01, 0.1, 0.2, 0.5, 1.0],
    #             'binarize': [0.0,0.05, 0.1, 0.3, 0.6, 1.0],
    #             'fit_prior': [True,False],
    #             'class_prior': [None]}
    #
    # CV = GridSearchCV(bernoulli, param_grid, refit = True, verbose = 3, n_jobs=-1)
    # CV.fit(TrainX, TrainY)
    #
    # optimalAlpha = CV.best_estimator_.alpha
    # optimalBinarize = CV.best_estimator_.binarize
    # optimalFitPrior = CV.best_estimator_.fit_prior
    # print("Optimal Alpha: ",optimalAlpha,", Optimal Binarize",optimalBinarize,", Optimal fit prior",optimalFitPrior,", Best score: ",CV.best_score_)

    # [2] Set optimal params.
    optimalAlpha = 0.0001
    optimalBinarize = 0
    optimalFitPrior = True
    print("Using Bernoulli estimator: Optimal Alpha: ",optimalAlpha,", Optimal Binarize",optimalBinarize,", Optimal fit prior",optimalFitPrior)

    # Choose best bernoulli, train and predict.
    best_Bernoulli = BernoulliNB(alpha=optimalAlpha,binarize=optimalBinarize,fit_prior=optimalFitPrior)
    best_Bernoulli.fit(TrainX, TrainY)
    predictions_Bernoulli = best_Bernoulli.predict(TestX)
    classificationReport = classification_report(y_pred = predictions_Bernoulli, y_true = TestY, output_dict = True)
    print("Predicted values: ",predictions_Bernoulli)

    sys.stdout = open("test-output-functional.csv", 'w')
    header = "Lower Case,Remove Stop Words,Stem,Test Size,Max Features,N-gram Range,TF,IDF,TFIDF,Accuracy"
    header += getHeaderFunctionalOnly()
    print(header)

    ngramRange_compact = str(parametersList[0].ngramRange[0]) + '-' + str(parametersList[0].ngramRange[1])
    print(parametersList[0].lowerCaseFlag, parametersList[0].removeStopWordsFlag, parametersList[0].stemFlag, parametersList[0].testSize,
                    parametersList[0].maxFeatures, ngramRange_compact, parametersList[0].tfidfFlags[0], parametersList[0].tfidfFlags[1],
                    parametersList[0].tfidfFlags[2], accuracy_score(y_pred = predictions_Bernoulli, y_true = TestY)*100,
                    classificationReport['Functional']['precision'], classificationReport['Functional']['recall'],classificationReport['Functional']['f1-score'], classificationReport['Functional']['support'],
                    classificationReport['Code']['precision'], classificationReport['Code']['recall'],classificationReport['Code']['f1-score'], classificationReport['Code']['support'],
                    classificationReport['IDE']['precision'], classificationReport['IDE']['recall'],classificationReport['IDE']['f1-score'], classificationReport['IDE']['support'],
                    classificationReport['General']['precision'], classificationReport['General']['recall'],classificationReport['General']['f1-score'], classificationReport['General']['support'],
                    classificationReport['Notice']['precision'], classificationReport['Notice']['recall'],classificationReport['Notice']['f1-score'], classificationReport['Notice']['support'],
                     sep=',')

    sys.stdout = original_stdout # Return to console
    print("Prediction stats are in: test-output-functional.csv")
    print("-----------------------------------  TEST END  -----------------------------------")
