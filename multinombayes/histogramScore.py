#manual preparation steps
#replace "(1, 1)" => 1.1 
#replace "(1, 1)" => 1.1 
#replace "(1, 1)" => 1.1 

import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import pandas as pd
my_data = genfromtxt( filepath, delimiter=',')

INDEX_MAX_FEATURES = 3
INDEX_ALPHA = 7
INDEX_ACCURACY=8
INDEX_F1SCORE = 11
INDEX_N_GRAM = 4

# f1score = []
# f1score.append(my_data[my_data[:,INDEX_MAX_FEATURES] == 1000])
# f1score.append(my_data[my_data[:,INDEX_MAX_FEATURES] == 5000])
# plt.subplot(211)
# plt.hist(f1score[0][:, INDEX_F1SCORE],bins=40, alpha=0.5, range=(0,1), label="1000 features") 
# plt.hist(f1score[1][:, INDEX_F1SCORE],bins=40, alpha=0.5, range=(0,1),  label="5000 features") 
# plt.legend(loc='upper left')
# plt.title("F1 score")

# plt.subplot(212)
# accu = []
# accu.append(my_data[my_data[:,INDEX_MAX_FEATURES] == 1000])
# accu.append(my_data[my_data[:,INDEX_MAX_FEATURES] == 5000])
# plt.hist(accu[0][:, INDEX_ACCURACY]/100,bins=40, alpha=0.5, range=(0,1), label="1000 features") 
# plt.hist(accu[1][:, INDEX_ACCURACY]/100,bins=40, alpha=0.5, range=(0,1),  label="5000 features") 
# plt.legend(loc='upper left')
# plt.title("Acuracy")


# f1score = []
# f1scoreMean = []
# alpha_range = [0.001, 0.01, 0.1, 0.25, 0.5, 1, 2]

# for alpha in alpha_range:
#     plt.hist(my_data[my_data[:,INDEX_ALPHA] == alpha][:, INDEX_F1SCORE],bins=100, density=True, histtype='step', cumulative=1, alpha=0.6, label=str(alpha)) 
#     f1scoreMean.append(np.mean(my_data[my_data[:,INDEX_ALPHA] == alpha][:, INDEX_F1SCORE]))

# plt.legend(loc='upper left')
# plt.figure()
# plt.plot(f1scoreMean) 
# plt.gca().set_xticks(range(0, len(f1scoreMean)))
# plt.gca().set_xticklabels(alpha_range)
# plt.legend(loc='upper left')


# f1score = []
# f1scoreMean = []
# rang = [1.1, 1.2, 1.3]

# for alpha in rang:
#    plt.hist(my_data[my_data[:,INDEX_N_GRAM] == alpha][:, INDEX_F1SCORE],bins=100, density=True, histtype='step', cumulative=1, alpha=0.5, label=str(alpha)) 
#    f1scoreMean.append(np.mean(my_data[my_data[:,INDEX_N_GRAM] == alpha][:, INDEX_F1SCORE]))

# plt.legend(loc='upper left')

# plt.figure()
# plt.plot(f1scoreMean) 
# plt.gca().set_xticks(range(0, len(f1scoreMean)))
# plt.gca().set_xticklabels(rang)


# accu = []
# accuMean = []
# rang = [1.1, 1.2, 1.3]

# for alpha in rang:
#    plt.hist(my_data[my_data[:,INDEX_N_GRAM] == alpha][:, INDEX_ACCURACY],bins=100, density=True, histtype='step', cumulative=1, alpha=0.5, label=str(alpha)) 
#    accuMean.append(np.mean(my_data[my_data[:,INDEX_N_GRAM] == alpha][:, INDEX_ACCURACY]))

# plt.legend(loc='upper left')

# plt.figure()
# plt.plot(accuMean) 
# plt.gca().set_xticks(range(0, len(accuMean)))
# plt.gca().set_xticklabels(rang)


boxplot = my_data.boxplot(column=['Macro F1-Score', 'Functional-Method F1-Score', 
'Functional-Module F1-Score', 'Functional-Inline F1-Score','Code F1-Score','IDE F1-Score','General F1-Score','Notice F1-Score','ToDo F1-Score'])

plt.show()