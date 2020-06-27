import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')
import preprocessing
import utilities
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
import gc

#Brankica 0
#Marija 1
#Stefan 2
iAm = 

# komanda za run iz foldera gde je ovaj fajl: python -W ignore logistic_regression.py

def compare_penalties(fileData):
    params = utilities.Parameters(
                            lowerCaseFlag=True, 
                            removeStopWordsFlag=True, 
                            stemFlag=False, 
                            maxFeatures=5000,
                            ngramRange=(1,2),
                            tfidfFlags=(False, True))
    Corpus, matrix, names = utilities.getInfoFromParameters(fileData, params)
    lrL1 = LogisticRegression(penalty = 'l1', solver='saga', tol = 0.01)
    lrL2 = LogisticRegression(penalty = 'l2')
    
    splits = 2
    outer_cv = StratifiedKFold(n_splits = splits, shuffle = True, random_state = 42)
    
    # CV for L1 estimator and L2 estimator which returns f1 and accuracy and we will compare it
    scoring = ['accuracy', 'f1_macro']
    scoresL2 = cross_validate(lrL2, X=matrix, y=Corpus['Class'], scoring = scoring, cv = outer_cv)
    
    scoresL1 = cross_validate(lrL1, X=matrix, y=Corpus['Class'], scoring = scoring, cv = outer_cv)
    
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
    #L1 accuracy:  0.9404406273338313  - L2 accuracy:  0.9432412247946229
    #L1 F1:  0.800465164204681  - L2 F1:  0.8037180312377004
    
if __name__ == "__main__":

    # Construct parameters.
    parametersList = list()
    for lowerCaseFlag in [True, False]:
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
    
    
    
    #iterToStart = 144 + 14 #158
    iterToStart = 288 # Skip all
    if iAm == 0:
        iterToStart = 0
    elif iAm == 1:
        iterToStart = 72
    elif iAm == 2:
        iterToStart = 144 + 72
    
    cnt = 0
    comparePenaltiesFlag = False
    # Go through all of the input files and configurations and export the results to a .csv file.
    for input_file, output_file, functionalOnlyFlag in [("../input.txt", "output-help.csv", False), ("../input-functional.txt", "output-help-functional.csv", True)]:
         with open(output_file, 'w') as output:
            print(utilities.getHeader(functionalOnlyFlag), file=output)

            fileData = preprocessing.read_file(input_file)
            if comparePenaltiesFlag:
                compare_penalties(fileData)
                import sys
                sys.exit()
            
            print("Processing started")
            for parameters in parametersList:
                if (cnt<iterToStart):
                    cnt = cnt + 1
                    continue
                # Find optimal hyperparameter C.
                Corpus, matrix, names = utilities.getInfoFromParameters(fileData, parameters)

                param_grid = {'C': [0.1, 1, 10]}

                inner_cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42)
                outer_cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

                # Inner CV.
                lr = LogisticRegression()
                gsLr = GridSearchCV(lr, param_grid, cv=inner_cv, scoring='f1_macro', n_jobs = -1)
                
                # Delete object from previous iteration
                gc.collect()
                len(gc.get_objects())
                # Outer CV. gs_lr.fit() gets called in cross_validate.
                cross_validate(gsLr, X=matrix, y=Corpus['Class'], scoring = utilities.scoringFunction, cv = outer_cv)

                utilities.printAverageValuesOfClassificationReportList(output, parameters, functionalOnlyFlag) 
                print(cnt)
                cnt = cnt + 1