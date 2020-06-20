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

    print("----------------------------------- PROGRAM START -----------------------------------")

    original_stdout = sys.stdout

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
    print("ParamsList created.\n")

    for input_file, output_file, is_functional in [("../input-functional.txt", "output-functional.csv", True), ("../input.txt", "output.csv", False)]:
         with open(output_file, 'w') as output_file_print_target:
            print("Using ",input_file, ", stats will be in ", output_file)
            fileData = preprocessing.read_file(input_file)

            # Print header in output file.
            header = "Lower Case,Remove Stop Words,Stem,Test Size,Max Features,N-gram Range,TF,TFIDF,Accuracy"
            if (is_functional):
                header += getHeaderFunctionalOnly()
            else:
                header += getHeaderAll()

            sys.stdout = output_file_print_target   # Change the standard output to the file we created.
            print(header)
            sys.stdout = original_stdout            # Reset the standard output to its original value

            for parameters in parametersList:
                # # For test, leave this comment for fast testing
                # if parameters != parametersList[0]:
                #     #print("Skip param...")
                #     continue

                print("Selected file processing param:")
                print("", "LowerCase", "RemoveStopWords", "Stem", "TestSize", "MaxFeatures", "N-gramRange", "TFIDF[0]", "TFIDF[1]", "TFIDF[2]", sep='\t\t\t')
                print("", parameters.lowerCaseFlag, parameters.removeStopWordsFlag, parameters.stemFlag, parameters.testSize, parameters.maxFeatures, parameters.ngramRange, parameters.tfidfFlags[0], parameters.tfidfFlags[1], parameters.tfidfFlags[2], sep='\t\t\t')

                Corpus, X, names = getInfoFromParameters(fileData, parameters)
                Y = Corpus["Class"]
                # Divide in Train and Test
                TrainX, TestX, TrainY, TestY = model_selection.train_test_split(X, Y,test_size=parameters.testSize)

                bernoulli = BernoulliNB()
                bernoulli.fit(TrainX,TrainY)

                print("Search for best estimator params...")
                # [1]Slow - Find optimal params
                param_grid = {'alpha': [0.0001,0.001, 0.01, 0.1, 0.2, 0.5, 1.0],
                            'binarize': [0.0,0.05, 0.1, 0.3, 0.6, 1.0],
                            'fit_prior': [True,False],
                            'class_prior': [None]}

                CV = GridSearchCV(bernoulli, param_grid, refit = True, verbose = 3, n_jobs=-1)
                CV.fit(TrainX, TrainY)

                optimalAlpha = CV.best_estimator_.alpha
                optimalBinarize = CV.best_estimator_.binarize
                optimalFitPrior = CV.best_estimator_.fit_prior
                print("Optimal Alpha: ",optimalAlpha,", Optimal Binarize",optimalBinarize,", Optimal fit prior",optimalFitPrior,", Best score: ",CV.best_score_)

                # # [2]Fast - Set default optimal params.
                # optimalAlpha = 0.0001
                # optimalBinarize = 0
                # optimalFitPrior = True
                # print("\tUsing Bernoulli estimator: Optimal Alpha: ",optimalAlpha,", Optimal Binarize",optimalBinarize,", Optimal fit prior",optimalFitPrior)

                # Choose best bernoulli, train and predict.
                best_Bernoulli = BernoulliNB(alpha=optimalAlpha,binarize=optimalBinarize,fit_prior=optimalFitPrior)
                best_Bernoulli.fit(TrainX, TrainY)
                print("Train...")
                predictions_Bernoulli = best_Bernoulli.predict(TestX)
                classificationReport = classification_report(y_pred = predictions_Bernoulli, y_true = TestY, output_dict = True)
                print("Predicted values: ",predictions_Bernoulli)

                sys.stdout = output_file_print_target # Change the standard output to the file we created.
                ngramRange_compact = "(" + str(parameters.ngramRange[0]) + '-' + str(parameters.ngramRange[1]) + ")"
                if (is_functional):
                    print(parameters.lowerCaseFlag, parameters.removeStopWordsFlag, parameters.stemFlag, parameters.testSize,
                                    parameters.maxFeatures, ngramRange_compact, parameters.tfidfFlags[0], parameters.tfidfFlags[1],
                                    accuracy_score(y_pred = predictions_Bernoulli, y_true = TestY)*100,
                                    classificationReport['Functional']['precision'], classificationReport['Functional']['recall'],classificationReport['Functional']['f1-score'], classificationReport['Functional']['support'],
                                    classificationReport['Code']['precision'], classificationReport['Code']['recall'],classificationReport['Code']['f1-score'], classificationReport['Code']['support'],
                                    classificationReport['IDE']['precision'], classificationReport['IDE']['recall'],classificationReport['IDE']['f1-score'], classificationReport['IDE']['support'],
                                    classificationReport['General']['precision'], classificationReport['General']['recall'],classificationReport['General']['f1-score'], classificationReport['General']['support'],
                                    classificationReport['Notice']['precision'], classificationReport['Notice']['recall'],classificationReport['Notice']['f1-score'], classificationReport['Notice']['support'],
                                    sep=',')
                else:
                    print(parameters.lowerCaseFlag, parameters.removeStopWordsFlag, parameters.stemFlag, parameters.testSize,
                                    parameters.maxFeatures, ngramRange_compact, parameters.tfidfFlags[0], parameters.tfidfFlags[1],
                                    accuracy_score(y_pred = predictions_Bernoulli, y_true = TestY)*100,
                                    classificationReport['Functional-Method']['precision'], classificationReport['Functional-Method']['recall'],classificationReport['Functional-Method']['f1-score'], classificationReport['Functional-Method']['support'],
                                    classificationReport['Functional-Module']['precision'], classificationReport['Functional-Module']['recall'],classificationReport['Functional-Module']['f1-score'], classificationReport['Functional-Module']['support'],
                                    classificationReport['Functional-Inline']['precision'], classificationReport['Functional-Inline']['recall'],classificationReport['Functional-Inline']['f1-score'], classificationReport['Functional-Inline']['support'],
                                    classificationReport['Code']['precision'], classificationReport['Code']['recall'],classificationReport['Code']['f1-score'], classificationReport['Code']['support'],
                                    classificationReport['IDE']['precision'], classificationReport['IDE']['recall'],classificationReport['IDE']['f1-score'], classificationReport['IDE']['support'],
                                    classificationReport['General']['precision'], classificationReport['General']['recall'],classificationReport['General']['f1-score'], classificationReport['General']['support'],
                                    classificationReport['Notice']['precision'], classificationReport['Notice']['recall'],classificationReport['Notice']['f1-score'], classificationReport['Notice']['support'],
                                    sep=',')
                sys.stdout = original_stdout # Reset the standard output to its original value.

            # END: for parameters in parametersList:
            print("Prediction stats are appended successfully in: ",output_file)
            print("\n")

         # END: for (input_file, output_file)

    print("-----------------------------------  PROGRAM END  -----------------------------------")
