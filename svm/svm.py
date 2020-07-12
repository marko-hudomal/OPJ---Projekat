import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

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

def compare_penalties(fileData):
    params = utilities.Parameters(
                            lowerCaseFlag=False, 
                            removeStopWordsFlag=False, 
                            stemFlag=False, 
                            maxFeatures=7363,
                            ngramRange=(1,1),
                            tfidfFlags=(False, False))

    svmL1 = svm.LinearSVC(penalty = 'l1', dual=False)
    svmL2 = svm.LinearSVC(penalty = 'l2')

    Corpus, pipelineL1 = utilities.getInfoFromParameters(fileData, params, svmL1)
    Corpus, pipelineL2 = utilities.getInfoFromParameters(fileData, params, svmL2)
    
    splits = 10
    outer_cv = StratifiedKFold(n_splits = splits, shuffle = True, random_state = 42)
    
    # CV for L1 estimator and L2 estimator which returns the F1-score and Accuracy.
    scoring = ['accuracy', 'f1_macro']
    scoresL2 = cross_validate(pipelineL2, X=Corpus[preprocessing.COMMENT], y=Corpus[preprocessing.CLASS], scoring = scoring, cv = outer_cv)
    scoresL1 = cross_validate(pipelineL1, X=Corpus[preprocessing.COMMENT], y=Corpus[preprocessing.CLASS], scoring = scoring, cv = outer_cv)
    
    for i in range(1,splits):
        scoresL1['test_accuracy'][0] += scoresL1['test_accuracy'][i]
        scoresL1['test_f1_macro'][0] += scoresL1['test_f1_macro'][i]
        scoresL2['test_accuracy'][0] += scoresL2['test_accuracy'][i]
        scoresL2['test_f1_macro'][0] += scoresL2['test_f1_macro'][i]
        
    scoresL1['test_accuracy'][0] /= splits
    scoresL1['test_f1_macro'][0] /= splits
    scoresL2['test_accuracy'][0] /= splits
    scoresL2['test_f1_macro'][0] /= splits
    
    print("L1 accuracy: ", scoresL1['test_accuracy'][0], " - L2 accuracy: " ,scoresL2['test_accuracy'][0])
    print("L1 F1: ", scoresL1['test_f1_macro'][0], " - L2 F1: " ,scoresL2['test_f1_macro'][0])
    # 1) All
    # L1 accuracy:  0.9575232207310785  - L2 accuracy:  0.9578971389550845
    # L1 F1:  0.8920203274995544  - L2 F1:  0.8922921461577517
    # 2) Functional-Only
    # L1 accuracy:  0.9792756281617125  - L2 accuracy:  0.9783424439427512
    # L1 F1:  0.9011988484009942  - L2 F1:  0.9067962967805394

if __name__ == "__main__":

    # Construct parameters.
    parametersList = list()

    for lowerCaseFlag in [False, True]:
        for removeStopWordsFlag in [False, True]:
            for stemFlag in [False, True]:
                    for maxFeatures in [1000, 5000, 7363]:
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
            # compare_penalties(fileData)

            for parameters in parametersList:
                
                # Preprocess data with current parameters.
                svm_clf = svm.SVC(kernel='linear')
                Corpus, pipeline = utilities.getInfoFromParameters(fileData, parameters, svm_clf)

                # Find optimal hyperparameter C using nested cross-validation.
                # [0.1, 1, 10, 100]
                param_grid = {'clf__C': [0.1, 1, 10],
                            'clf__kernel': ['linear']}

                inner_cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42)
                outer_cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

                # Inner CV.
                gsPipeline = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='f1_macro', n_jobs  = -1, refit = True)

                # Outer CV. gs_lr.fit() gets called in cross_validate.
                cross_validate(gsPipeline, X=Corpus[preprocessing.COMMENT], y=Corpus[preprocessing.CLASS], scoring = utilities.scoringFunction, cv = outer_cv, return_train_score = False)

                utilities.printAverageValuesOfClassificationReportList(output, parameters, functionalOnlyFlag)
