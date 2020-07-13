

import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import pandas as pd
my_data = genfromtxt( filepath, delimiter=',')

INDEX_MAX_FEATURES = 3
INDEX_ALPHA = 7
INDEX_ACCURACY=9
INDEX_F1SCORE = 12
INDEX_N_GRAM = 4

f1score = []
f1score.append(my_data[my_data[:,INDEX_MAX_FEATURES] == 7363])
f1score.append(my_data[my_data[:,INDEX_MAX_FEATURES] == 5000])
f1score.append(my_data[my_data[:,INDEX_MAX_FEATURES] == 1000])
plt.subplot(211)
plt.hist(f1score[0][:, INDEX_F1SCORE],bins=50, alpha=0.4, range=(0.6,0.95), label="7363 features F1-Score") 
plt.hist(f1score[1][:, INDEX_F1SCORE],bins=50, alpha=0.4, range=(0.6,0.95), label="5000 features F1-Score") 
plt.hist(f1score[2][:, INDEX_F1SCORE],bins=50, alpha=0.4, range=(0.6,0.95), label="1000 features F1-Score") 

plt.legend(loc='upper left')
plt.title("F1 score")

plt.subplot(212)
accu = []
accu.append(my_data[my_data[:,INDEX_MAX_FEATURES] == 7363])
accu.append(my_data[my_data[:,INDEX_MAX_FEATURES] == 5000])
accu.append(my_data[my_data[:,INDEX_MAX_FEATURES] == 1000])
plt.hist(accu[0][:, INDEX_ACCURACY]/100,bins=50, alpha=0.4, range=(0.6,0.95), label="7363 features Accuracy")
plt.hist(accu[1][:, INDEX_ACCURACY]/100,bins=50, alpha=0.4, range=(0.6,0.95), label="5000 features Accuracy") 
plt.hist(accu[2][:, INDEX_ACCURACY]/100,bins=50, alpha=0.4, range=(0.6,0.95), label="1000 features Accuracy")  
plt.legend(loc='upper left')
plt.xlabel('Интервали')
plt.ylabel('број комбинација параметара које упадају у одређени интервал')
plt.title("Acuracy")


# f1score = []
# f1scoreMean = []
# f1scoreMax= []
# f1scoreMin= []
# max_features = [7363, 5000, 1000]

# for alpha in max_features:
#    plt.hist(my_data[my_data[:,INDEX_MAX_FEATURES] == alpha][:, INDEX_F1SCORE], bins=20, density=True, alpha=0.4, label=str(alpha)) 
#    f1scoreMean.append(np.mean(my_data[my_data[:,INDEX_MAX_FEATURES] == alpha][:, INDEX_F1SCORE]))
#    f1scoreMax.append(np.max(my_data[my_data[:,INDEX_MAX_FEATURES] == alpha][:, INDEX_F1SCORE]))
#    f1scoreMin.append(np.min(my_data[my_data[:,INDEX_MAX_FEATURES] == alpha][:, INDEX_F1SCORE]))

# plt.legend(loc='upper left')
# plt.figure()
# plt.plot(f1scoreMean, label='mean') 
# plt.plot(f1scoreMax, label='max')
# plt.plot(f1scoreMin, label='min')
# plt.xlabel('max_features')
# plt.ylabel('Macro F1-Score')
# plt.title('Macro F1-Score in function of max_features')
# plt.gca().set_xticks(range(0, len(f1scoreMean)))
# plt.gca().set_xticklabels(max_features)
# plt.legend(loc='upper right')


# f1score = []
# f1scoreMean = []
# f1scoreMax= []
# alpha_range = [0.00001, 0.001, 1]

# for alpha in alpha_range:
#     plt.hist(my_data[my_data[:,INDEX_ALPHA] == alpha][:, INDEX_F1SCORE], bins=20, density=True, alpha=0.6, label=str(alpha)) 
#     f1scoreMean.append(np.mean(my_data[my_data[:,INDEX_ALPHA] == alpha][:, INDEX_F1SCORE]))
#     f1scoreMax.append(np.max(my_data[my_data[:,INDEX_ALPHA] == alpha][:, INDEX_F1SCORE]))

# plt.legend(loc='upper left')
# plt.figure()
# plt.plot(f1scoreMean, label='mean') 
# plt.plot(f1scoreMax, label='max')
# plt.xlabel('alpha')
# plt.ylabel('Macro F1-Score')
# plt.title('Macro F1-Score in function of alpha')
# plt.gca().set_xticks(range(0, len(f1scoreMean)))
# plt.gca().set_xticklabels(alpha_range)
# plt.legend(loc='upper right')

# f1score = []
# f1scoreMean = []
# f1scoreMax= []
# rang = [1.1, 1.2, 1.3]

# for alpha in rang:
#    plt.hist(my_data[my_data[:,INDEX_N_GRAM] == alpha][:, INDEX_F1SCORE],bins=100, alpha=0.5, density=True, label=str(alpha)) 
#    f1scoreMean.append(np.mean(my_data[my_data[:,INDEX_N_GRAM]==alpha ][:, INDEX_F1SCORE]))
#    f1scoreMax.append(np.max(my_data[my_data[:,INDEX_N_GRAM]==alpha][:, INDEX_F1SCORE]))

# plt.legend(loc='upper left')
# plt.figure()
# plt.plot(f1scoreMean, label='mean') 
# plt.plot(f1scoreMax, label='max')
# plt.xlabel('N-grams')
# plt.ylabel('Macro F1-Score')
# plt.title('Macro F1-Score in function of N-grams')
# plt.gca().set_xticks(range(0, len(f1scoreMean)))
# plt.gca().set_xticklabels(rang)
# plt.legend(loc='upper left')


plt.show()