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
from sklearn import model_selection, naive_bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate

import sys

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime

class Parameters :
    def __init__(self, lowerCaseFlag, removeStopWordsFlag, stemFlag, maxFeatures, ngramRange, tfidfFlags, alpha):
        self.lowerCaseFlag = lowerCaseFlag
        self.removeStopWordsFlag = removeStopWordsFlag
        self.stemFlag = stemFlag
        self.maxFeatures = maxFeatures
        self.ngramRange = ngramRange
        self.tfidfFlags = tfidfFlags
        self.alpha = alpha

def getInfoFromParameters(input_file, parameters):

    Corpus = preprocessing.process_data(input_file, to_lower_case=parameters.lowerCaseFlag, remove_stop_words=parameters.removeStopWordsFlag, stem=parameters.stemFlag)
    matrix, names = preprocessing.vectorize(Corpus, max_features=parameters.maxFeatures, ngram_range=parameters.ngramRange, tf=parameters.tfidfFlags[0], tfidf=parameters.tfidfFlags[1])
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
    # called once for each fold cross validation on test set 
    global classificationReportList

    predictions = estimator.predict(x)
    classificationReport = classification_report(y_pred = predictions, y_true = y, output_dict = True)
    classificationReport['allClasses'] = {  'f1score' : f1_score(y_pred = predictions, y_true = y, average="macro")*100,
                                            'accu' : accuracy_score(y_pred = predictions, y_true = y)*100, }

    classificationReportList.append(classificationReport)

    return 1

def printAverageValuesOfClassificationReportList(outputStream, outputFileName, parameters, functionalMultipleSubCategories):

    global classificationReportList
    resultDictionary = {}

    commentClassArray = []
    if (functionalMultipleSubCategories):
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
    
    # average out score on 10 fold cross validation reported scores per one hyper-parameters combination
    # f1score and accu are  manually computed
    f1score = 0
    for i in range(0, len(classificationReportList)):
        f1score = f1score + classificationReportList[i]["allClasses"]["f1score"]
    
    f1score = f1score / len(classificationReportList)

    accu = 0
    for i in range(0, len(classificationReportList)):
        accu = accu + classificationReportList[i]["allClasses"]["accu"]
    
    accu = accu / len(classificationReportList)


    if (functionalMultipleSubCategories):
        print(parameters.lowerCaseFlag, parameters.removeStopWordsFlag, parameters.stemFlag, parameters.maxFeatures,  "\"" + str(parameters.ngramRange) + "\"", parameters.tfidfFlags[0], parameters.tfidfFlags[1], parameters.alpha, f1score, accu,
        resultDictionary['Functional-Method']['precision'], resultDictionary['Functional-Method']['recall'], resultDictionary['Functional-Method']['f1-score'], resultDictionary['Functional-Method']['support'],
        resultDictionary['Functional-Module']['precision'], resultDictionary['Functional-Module']['recall'], resultDictionary['Functional-Module']['f1-score'], resultDictionary['Functional-Module']['support'],
        resultDictionary['Functional-Inline']['precision'], resultDictionary['Functional-Inline']['recall'], resultDictionary['Functional-Inline']['f1-score'], resultDictionary['Functional-Inline']['support'],
        resultDictionary['Code']['precision'], resultDictionary['Code']['recall'], resultDictionary['Code']['f1-score'], resultDictionary['Code']['support'],
        resultDictionary['IDE']['precision'], resultDictionary['IDE']['recall'], resultDictionary['IDE']['f1-score'], resultDictionary['IDE']['support'],
        resultDictionary['General']['precision'], resultDictionary['General']['recall'], resultDictionary['General']['f1-score'], resultDictionary['General']['support'],
        resultDictionary['Notice']['precision'], resultDictionary['Notice']['recall'], resultDictionary['Notice']['f1-score'], resultDictionary['Notice']['support'],
        resultDictionary['ToDo']['precision'], resultDictionary['ToDo']['recall'], resultDictionary['ToDo']['f1-score'], resultDictionary['ToDo']['support'], sep=',', file=outputStream)
    else:
        print(outputStream, parameters.lowerCaseFlag, parameters.removeStopWordsFlag, parameters.stemFlag, parameters.maxFeatures, "'" + str(parameters.ngramRange) + "'",  parameters.tfidfFlags[0], parameters.tfidfFlags[1], parameters.alpha, f1score, accu,
        resultDictionary['Functional']['precision'], resultDictionary['Functional']['recall'], resultDictionary['Functional']['f1-score'], resultDictionary['Functional']['support'],
        resultDictionary['Code']['precision'], resultDictionary['Code']['recall'], resultDictionary['Code']['f1-score'], resultDictionary['Code']['support'],
        resultDictionary['IDE']['precision'], resultDictionary['IDE']['recall'], resultDictionary['IDE']['f1-score'], resultDictionary['IDE']['support'],
        resultDictionary['General']['precision'], resultDictionary['General']['recall'], resultDictionary['General']['f1-score'], resultDictionary['General']['support'],
        resultDictionary['Notice']['precision'], resultDictionary['Notice']['recall'], resultDictionary['Notice']['f1-score'], resultDictionary['Notice']['support'],
        resultDictionary['ToDo']['precision'], resultDictionary['ToDo']['recall'], resultDictionary['ToDo']['f1-score'], resultDictionary['ToDo']['support'], sep=',', file=outputStream)

    outputStream.flush()
    classificationReportList = []

if __name__ == "__main__":

    # Construct parameters.
    parametersList = list()

    lowerCaseFlag = True

    for removeStopWordsFlag in [False, True]:
        for stemFlag in [False, True]:
            for maxFeatures in [1000, 5000]:
                for ngramRange in [(1, 1), (1, 2), (1, 3)]:
                    for tfidfFlags in [(False, False), (True, False), (False, True)]:
                        for  alpha_value in [2, 1, 0.5, 0.25, 0.1]:
                            parametersList.append(Parameters(
                                lowerCaseFlag, 
                                removeStopWordsFlag, 
                                stemFlag, 
                                maxFeatures,
                                ngramRange,
                                tfidfFlags,
                                alpha_value)
                            )

    # Save a reference to the original standard output
    original_stdout = sys.stdout
    cnt = 0

    # Go through all of the input files and configurations and export the results to a .csv file.
    for input_file, output_file_path, functionalMultipleSubCategories in [("svi sredjeni.txt", "outputNBdirectAlpha.csv", True), ("svi sredjeni functional.txt", "outputNBdirectAlphaFunctional.csv", False)]:
         with open(output_file_path, 'w') as output:
            #sys.stdout = output
            
            header = "Lower Case,Remove Stop Words,Stem,Max Features,N-gram Range,TF,TFIDF,alpha, F1score, accu "
            if (functionalMultipleSubCategories):
                header += getHeaderAll()
            else:
                header += getHeaderFunctionalOnly()
            print(header, file=output)
            output.flush()

            fileData = preprocessing.read_file(input_file)

            for parameters in parametersList:
                print(cnt, ' / ', len(parametersList))
                # datetime object containing current date and time
                print(">>>>>>>>>>>>>>>>>>>>> get info start. now =",  datetime.now())           
                Corpus, matrix, names = getInfoFromParameters(fileData, parameters)

                outer_cv = KFold(n_splits = 10, shuffle = True, random_state = 42)

                classifier = MultinomialNB(alpha = parameters.alpha)

                # Outer CV. Multinomial.fit() gets called in cross_validate.
                print(">>>>>>>>>>>>>>>>>>>>> c_v start. now =",  datetime.now())
                cross_validate(classifier, X=matrix, y=Corpus['Class'], scoring = scoringFunction, cv = outer_cv, return_train_score = False)
                print("<<<<<<<<<<<<<<<<<<<<<< c_v end. now =",  datetime.now())

                # Print to csv
                printAverageValuesOfClassificationReportList(output, output_file_path, parameters, functionalMultipleSubCategories)

                cnt = cnt + 1

                # Print progress info to screen for visualization
                print(vars(parameters))           

            sys.stdout = original_stdout