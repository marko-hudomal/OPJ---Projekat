# Add parent folder to path for importing preprocessing and utilities
import sys
sys.path.insert(0,'..')

# Import our modules
import preprocessing
import utilities

# Import libraries
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
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_validate

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime

if __name__ == "__main__":

    # Construct parameters.
    parametersList = list()

    lowerCaseFlag = True

    for removeStopWordsFlag in [False, True]:
        for stemFlag in [False, True]:
            for maxFeatures in [1000, 5000]:
                for ngramRange in [(1, 1), (1, 2), (1, 3)]:
                    for tfidfFlags in [(False, False), (True, False), (False, True)]:
                        for  alpha_value in [2, 1, 0.5, 0.25, 0.1, 0.01, 0.001]:
                            parametersList.append(utilities.Parameters(
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
    for input_file, output_file_path, singleFunctionalClass in [("../svi sredjeni.txt", "output/outputNBdirectAlpha.csv", False), ("../svi sredjeni functional.txt", "output/outputNBdirectAlphaFunctional.csv", True)]:
         with open(output_file_path, 'w') as output:
            print(utilities.getHeader(singleFunctionalClass), file=output)
            output.flush()

            fileData = preprocessing.read_file(input_file)

            for parameters in parametersList:
                print(cnt, ' / ', len(parametersList))
                # datetime object containing current date and time
                print(">>>>>>>>>>>>>>>>>>>>> get info start. now =",  datetime.now())           
                Corpus, matrix, names = utilities.getInfoFromParameters(fileData, parameters)

                outer_cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

                classifier = MultinomialNB(alpha = parameters.alphaNaiveBayes)

                # Outer CV. Multinomial.fit() gets called in cross_validate.
                print(">>>>>>>>>>>>>>>>>>>>> c_v start. now =",  datetime.now())
                cross_validate(classifier, X=matrix, y=Corpus['Class'], scoring = utilities.scoringFunction, cv = outer_cv, return_train_score = False)
                print("<<<<<<<<<<<<<<<<<<<<<< c_v end. now =",  datetime.now())

                # Print to csv
                utilities.printAverageValuesOfClassificationReportList(output, parameters, singleFunctionalClass)
                output.flush()

                cnt = cnt + 1

                # Print progress info to screen for visualization
                print(vars(parameters))           

            sys.stdout = original_stdout