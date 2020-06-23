import preprocessing
import utilities
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
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

if __name__ == "__main__":

    # Construct parameters.
    parametersList = list()

    for lowerCaseFlag in [True]:
        for removeStopWordsFlag in [False, True]:
            for stemFlag in [False, True]:
                    for maxFeatures in [1000, 5000]:
                        for ngramRange in [(1, 1), (1, 2), (1, 3)]:
                            for tfidfFlags in [(False, False), (True, False), (False, True)]:
                                parametersList.append(utilities.Parameters(
                                    lowerCaseFlag, 
                                    removeStopWordsFlag, 
                                    stemFlag, 
                                    maxFeatures,
                                    ngramRange,
                                    tfidfFlags)
                                )
    
    # Go through all of the input files and configurations and export the results to a .csv file.
    for input_file, output_file, functionalOnlyFlag in [("input.txt", "output.csv", False), ("input-functional.txt", "output-functional.csv", True)]:
         with open(output_file, 'w') as output:
            
            print(utilities.getHeader(functionalOnlyFlag), file=output)

            fileData = preprocessing.read_file(input_file)

            for parameters in parametersList:
                # Find optimal hyperparameter C.
                Corpus, matrix, names = utilities.getInfoFromParameters(fileData, parameters)

                # [0.1, 1, 10, 100]
                param_grid = {'C': [0.1, 1, 10],
                            'kernel': ['linear']}

                inner_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
                outer_cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

                # Inner CV.
                SVM = GridSearchCV(svm.SVC(), param_grid, refit = True, cv=inner_cv, scoring='f1_macro', n_jobs = 4)

                # Outer CV. SVM.fit() gets called in cross_validate.
                cross_validate(SVM, X=matrix, y=Corpus['Class'], scoring = utilities.scoringFunction, cv = outer_cv, return_train_score = False)

                utilities.printAverageValuesOfClassificationReportList(output, parameters, functionalOnlyFlag)
