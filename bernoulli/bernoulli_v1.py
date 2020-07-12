import preprocessing
import utilities
import warnings
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
from sklearn.model_selection import GridSearchCV, KFold,StratifiedKFold, cross_validate
from sklearn.naive_bayes import BernoulliNB

import sys



if __name__ == "__main__":

    print("----------------------------------- PROGRAM START -----------------------------------")
    start_time = time.time()
    original_stdout = sys.stdout

    warnings.filterwarnings('ignore')

    # Construct parameters.
    parametersList = list()

    for lowerCaseFlag in [False, True]:
        for removeStopWordsFlag in [False, True]:
            for stemFlag in [False, True]:
                    for maxFeatures in [1000, 5000, 7363]:
                        for ngramRange in [(1, 1), (1, 2), (1, 3)]:
                            for alpha in [0.00001, 0.001, 1]:
                                for binarize in [0.0]:
                                    for tfidfFlags in [(False, False)]:
                                        parametersList.append(utilities.Parameters(
                                            lowerCaseFlag,
                                            removeStopWordsFlag,
                                            stemFlag,
                                            maxFeatures,
                                            ngramRange,
                                            tfidfFlags,
                                            alpha,
                                            binarize)
                                )
    print("ParamsList created.\n")

    count_file = 0
    for input_file, output_file, is_functional in [("../input-functional.txt", "output-functional.csv", True), ("../input.txt", "output.csv", False)]:
         with open(output_file, 'w') as output_file_print_target:
            print("Using ",input_file, ", stats will be in ", output_file)
            fileData = preprocessing.read_file(input_file)

            # Print header in output file.
            header = utilities.getHeader(is_functional)

            sys.stdout = output_file_print_target   # Change the standard output to the file we created.
            print(header)
            sys.stdout = original_stdout            # Reset the standard output to its original value

            count = 0
            for parameters in parametersList:
                # # For test, leave this comment for fast testing
                # if parameters != parametersList[0]:
                #     #print("Skip param...")
                #     continue

                print(utilities.bcolors.WARNING + "***PROGRESS*** file: [",count_file,"/ 2 ], param: [", count,"/",parametersList.__len__(),"]" + utilities.bcolors.ENDC)

                print("Selected file processing param:")
                print("\tLowerCase: {0}| RemoveStopWords: {1}| Stem: {2}| MaxFeatures: {3}| N-gramRange: {4}| alpha: {5}| binarize: {6}".format(parameters.lowerCaseFlag, parameters.removeStopWordsFlag, parameters.stemFlag, parameters.maxFeatures, parameters.ngramRange, parameters.alphaNaiveBayes, parameters.binarizeNaiveBayes), sep='\t')

                # Choose best bernoulli, train and predict.
                best_Bernoulli = BernoulliNB(alpha = parameters.alphaNaiveBayes, binarize = parameters.binarizeNaiveBayes)

                Corpus, pipeline = utilities.getInfoFromParameters(fileData, parameters, best_Bernoulli)

                # Cross validation.
                outer_cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
                # Outer CV. best_Bernoulli.fit() gets called in cross_validate.
                cross_validate(pipeline, X=Corpus[preprocessing.COMMENT], y=Corpus[preprocessing.CLASS], scoring = utilities.scoringFunction, cv = outer_cv, return_train_score = False)

                # Print to output file.
                utilities.printAverageValuesOfClassificationReportList(output_file_print_target, parameters, is_functional)
                output_file_print_target. flush()

                count+=1

            # END: for parameters in parametersList:
            print("Prediction stats are appended successfully in: ",output_file)
            print("\n")
         count_file+=1

    # END: for (input_file, output_file)

    print("-----------------------------------  PROGRAM END  -----------------------------------")
    print("--- %s seconds ---" % (time.time() - start_time))
    
