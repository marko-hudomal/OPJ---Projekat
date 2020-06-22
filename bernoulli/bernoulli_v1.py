import preprocessing
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
    Corpus = preprocessing.process_data(input_file, to_lower_case=parameters.lowerCaseFlag, remove_stop_words=parameters.removeStopWordsFlag, stem=parameters.stemFlag)
    counts_by_comment, names = preprocessing.vectorize(Corpus, max_features=parameters.maxFeatures, ngram_range=parameters.ngramRange, tf=parameters.tfidfFlags[0], tfidf=parameters.tfidfFlags[1])

    return Corpus, counts_by_comment, names

classificationReportList = []
def scoringFunction(estimator, x, y):

    global classificationReportList

    predictions = estimator.predict(x)
    classificationReport = classification_report(y_pred = predictions, y_true = y, output_dict = True)
    classificationReportList.append([accuracy_score(y_pred = predictions, y_true = y)*100, classificationReport])

    return 1

def printAverageValuesOfClassificationReportList(outputFile, parameters, functionalOnlyFlag):

    original_stdout = sys.stdout
    sys.stdout = output_file_print_target # Change the standard output to the file we created.

    # [0] accuracy_score
    # [1] classification_report
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

    ngramRange_compact = "(" + str(parameters.ngramRange[0]) + '-' + str(parameters.ngramRange[1]) + ")"
    if (functionalOnlyFlag == False):
        print(parameters.lowerCaseFlag, parameters.removeStopWordsFlag, parameters.stemFlag, parameters.maxFeatures, ngramRange_compact, parameters.tfidfFlags[0], parameters.tfidfFlags[1], accuracy,
        resultDictionary['Functional-Method']['precision'], resultDictionary['Functional-Method']['recall'], resultDictionary['Functional-Method']['f1-score'], resultDictionary['Functional-Method']['support'],
        resultDictionary['Functional-Module']['precision'], resultDictionary['Functional-Module']['recall'], resultDictionary['Functional-Module']['f1-score'], resultDictionary['Functional-Module']['support'],
        resultDictionary['Functional-Inline']['precision'], resultDictionary['Functional-Inline']['recall'], resultDictionary['Functional-Inline']['f1-score'], resultDictionary['Functional-Inline']['support'],
        resultDictionary['Code']['precision'], resultDictionary['Code']['recall'], resultDictionary['Code']['f1-score'], resultDictionary['Code']['support'],
        resultDictionary['IDE']['precision'], resultDictionary['IDE']['recall'], resultDictionary['IDE']['f1-score'], resultDictionary['IDE']['support'],
        resultDictionary['General']['precision'], resultDictionary['General']['recall'], resultDictionary['General']['f1-score'], resultDictionary['General']['support'],
        resultDictionary['Notice']['precision'], resultDictionary['Notice']['recall'], resultDictionary['Notice']['f1-score'], resultDictionary['Notice']['support'],
        resultDictionary['ToDo']['precision'], resultDictionary['ToDo']['recall'], resultDictionary['ToDo']['f1-score'], resultDictionary['ToDo']['support'], sep=',')
    else:
        print(parameters.lowerCaseFlag, parameters.removeStopWordsFlag, parameters.stemFlag, parameters.maxFeatures, ngramRange_compact, parameters.tfidfFlags[0], parameters.tfidfFlags[1], accuracy,
        resultDictionary['Functional']['precision'], resultDictionary['Functional']['recall'], resultDictionary['Functional']['f1-score'], resultDictionary['Functional']['support'],
        resultDictionary['Code']['precision'], resultDictionary['Code']['recall'], resultDictionary['Code']['f1-score'], resultDictionary['Code']['support'],
        resultDictionary['IDE']['precision'], resultDictionary['IDE']['recall'], resultDictionary['IDE']['f1-score'], resultDictionary['IDE']['support'],
        resultDictionary['General']['precision'], resultDictionary['General']['recall'], resultDictionary['General']['f1-score'], resultDictionary['General']['support'],
        resultDictionary['Notice']['precision'], resultDictionary['Notice']['recall'], resultDictionary['Notice']['f1-score'], resultDictionary['Notice']['support'],
        resultDictionary['ToDo']['precision'], resultDictionary['ToDo']['recall'], resultDictionary['ToDo']['f1-score'], resultDictionary['ToDo']['support'], sep=',')

    classificationReportList = []
    sys.stdout = original_stdout # Reset the standard output to its original value.

def getHeader(is_functional):
    if (is_functional):
        result = ",Functional Precision,Functional Recall,Functional F1-Score,Functional Support"
    else:
        result = ",Functional-Method Precision,Functional-Method Recall,Functional-Method F1-Score,Functional-Method Support" + ",Functional-Module Precision,Functional-Module Recall,Functional-Module F1-Score,Functional-Module Support" + ",Functional-Inline Precision,Functional-Inline Recall,Functional-Inline F1-Score,Functional-Inline Support"

    result += ",Code Precision,Code Recall,Code F1-Score,Code Support" + ",IDE Precision,IDE Recall,IDE F1-Score,IDE Support" + ",General Precision,General Recall,General F1-Score,General Support" + ",Notice Precision,Notice Recall,Notice F1-Score,Notice Support"
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

if __name__ == "__main__":

    print("----------------------------------- PROGRAM START -----------------------------------")
    start_time = time.time()
    original_stdout = sys.stdout

    # Construct parameters.
    parametersList = list()

    for lowerCaseFlag in [True]:
        for removeStopWordsFlag in [False, True]:
            for stemFlag in [False, True]:
                    for maxFeatures in [1000, 5000]:
                        for ngramRange in [(1, 1), (1, 2), (1, 3)]:
                            for tfidfFlags in [(False, False), (True, True), (False, True), (True, False)]:
                                parametersList.append(Parameters(
                                    lowerCaseFlag,
                                    removeStopWordsFlag,
                                    stemFlag,
                                    maxFeatures,
                                    ngramRange,
                                    tfidfFlags)
                                )
    print("ParamsList created.\n")

    count_file = 0
    for input_file, output_file, is_functional in [("../input-functional.txt", "output-functional.csv", True), ("../input.txt", "output.csv", False)]:
         with open(output_file, 'w') as output_file_print_target:
            print("Using ",input_file, ", stats will be in ", output_file)
            fileData = preprocessing.read_file(input_file)

            # Print header in output file.
            header = "Lower Case,Remove Stop Words,Stem,Max Features,N-gram Range,TF,TFIDF,Accuracy"
            header += getHeader(is_functional)

            sys.stdout = output_file_print_target   # Change the standard output to the file we created.
            print(header)
            sys.stdout = original_stdout            # Reset the standard output to its original value

            count = 0
            for parameters in parametersList:
                # # For test, leave this comment for fast testing
                # if parameters != parametersList[0]:
                #     #print("Skip param...")
                #     continue

                print(bcolors.WARNING + "***PROGRESS*** file: [",count_file,"/ 2 ], param: [", count,"/",parametersList.__len__(),"]" + bcolors.ENDC)

                print("Selected file processing param:")
                print("\tLowerCase: {0}| RemoveStopWords: {1}| Stem: {2}| MaxFeatures: {3}| N-gramRange: {4}| TFIDF[0]: {5}| TFIDF[1]: {6}".format(parameters.lowerCaseFlag, parameters.removeStopWordsFlag, parameters.stemFlag, parameters.maxFeatures, parameters.ngramRange, parameters.tfidfFlags[0], parameters.tfidfFlags[1]), sep='\t')

                Corpus, X, names = getInfoFromParameters(fileData, parameters)
                Y = Corpus["Class"]
                print("Search for best estimator params...")
                # # [1]Slow - Find optimal params
                # param_grid = {'alpha': [0.0001,0.001, 0.01, 0.1, 0.2, 0.5, 1.0],
                #             'binarize': [0.0,0.05, 0.1, 0.3, 0.6, 1.0],
                #             'fit_prior': [True,False],
                #             'class_prior': [None]}
                #
                # CV = GridSearchCV(bernoulli, param_grid, refit = True, verbose = 1, n_jobs=-1)
                # CV.fit(TrainX, TrainY)
                #
                # optimalAlpha = CV.best_estimator_.alpha
                # optimalBinarize = CV.best_estimator_.binarize
                # optimalFitPrior = CV.best_estimator_.fit_prior
                # print("Optimal Alpha: ",optimalAlpha,", Optimal Binarize",optimalBinarize,", Optimal fit prior",optimalFitPrior,", Best score: ",CV.best_score_)

                # [2]Fast - Set default optimal params.
                optimalAlpha = 0.0001
                optimalBinarize = 0
                optimalFitPrior = True
                print("\tUsing Bernoulli estimator: Optimal Alpha: ",optimalAlpha,", Optimal Binarize",optimalBinarize,", Optimal fit prior",optimalFitPrior)

                # Choose best bernoulli, train and predict.
                best_Bernoulli = BernoulliNB(alpha=optimalAlpha,binarize=optimalBinarize,fit_prior=optimalFitPrior)

                # Cross validation.
                outer_cv = KFold(n_splits = 10, shuffle = True, random_state = 42)
                # Outer CV. best_Bernoulli.fit() gets called in cross_validate.
                cross_validate(best_Bernoulli, X=X, y=Y, scoring = scoringFunction, cv = outer_cv, return_train_score = False)

                # Print to output file.
                printAverageValuesOfClassificationReportList(output_file, parameters, is_functional)

                count+=1

            # END: for parameters in parametersList:
            print("Prediction stats are appended successfully in: ",output_file)
            print("\n")
         count_file+=1

    # END: for (input_file, output_file)

    print("-----------------------------------  PROGRAM END  -----------------------------------")
    print("--- %s seconds ---" % (time.time() - start_time))
