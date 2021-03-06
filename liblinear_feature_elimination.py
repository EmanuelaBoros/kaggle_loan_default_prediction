import pandas

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn import cross_validation

from sklearn.preprocessing import Imputer

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn import preprocessing as pre
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm, datasets

import statsmodels.api as sm

import numpy as np
import pylab as pl
from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn import metrics
import pickle

from sklearn.metrics import mean_absolute_error

from scipy.stats.stats import pearsonr, spearmanr
from sklearn.svm import SVR

import os.path

import subprocess
import os

from sklearn.naive_bayes import GaussianNB

liblinear_path = "/home/ema/Workspace/Tools/liblinear-1.94/"

def eliminate_features():

    use_sample = False
    
    if use_sample:
        train = pandas.read_csv('data/train_v2_sample_10k.csv')
#        test = pandas.read_csv('data/test_v2_sample_10k.csv')
        average_best_t = 0.148846575958
    else:
        train = pandas.read_csv('data/train_v2.csv')
#        test = pandas.read_csv('data/test_v2.csv')
         ### To use on full train set
        average_best_t = 0.164463473639
  
    
    train_loss = train.loss
    
    cols = set(train.columns)
    cols.remove('loss')
    cols = list(cols)
    train = train[cols] 
    
    column_names = train.columns.values.tolist()

    
#    train = train[['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431', 'f757']]
    

    imp = Imputer()
    imp.fit(train)
    
    train = imp.transform(train)
#    test = imp.transform(test)
    
    train=pre.StandardScaler().fit_transform(train)
#    test=pre.StandardScaler().fit_transform(test)    
    
    train_loss_array_libsvm = train_loss.apply(lambda x: -1 if x>0 else 1).values
#    b = np.delete(train,0,1)
#    c = np.delete(train,1,1)
#    print b.shape[1]
#    print c.shape
    
    
    best_acc = 91.3437
    best_eliminated_features = []
    
    best_features = [18, 289, 290, 17, 402, 19, 560, 16, 287, 310, 403]
    selected_train = train[:,best_features]    

    os.chdir(liblinear_path)
    train_command = "./train -s 5 -c 0.01 -v 5 -e 0.001 /home/ema/Workspace/Projects/Kaggle/Loan_Default_Prediction/data/train_tmp.liblinear"
    datasets.dump_svmlight_file(selected_train, train_loss_array_libsvm, "/home/ema/Workspace/Projects/Kaggle/Loan_Default_Prediction/data/train_selected_f.liblinear", zero_based=False, comment=None, query_id=None)
    generation = 0    
#    while generation < 1:        
#        eliminated_features = best_eliminated_features
#        for i in range(0,train.shape[1]):            
#            eliminated_features = best_eliminated_features + [i] 
#            reduced_train = train
#            for f in eliminated_features:
#                reduced_train = np.delete(reduced_train,f,1)
#            print reduced_train.shape
#            
##            datasets.dump_svmlight_file(reduced_train, train_loss_array_libsvm, "/home/ema/Workspace/Projects/Kaggle/Loan_Default_Prediction/data/train_tmp.liblinear", zero_based=False, comment=None, query_id=None)
#
#            print "Training"
#            proc = subprocess.Popen([train_command], stdout=subprocess.PIPE, shell=True)
#            (out, err) = proc.communicate()
#            print out.split("\n")[-2]
#            score = float(out.split("\n")[-2].split("=")[1].strip()[:-1])
#            
#            os.remove("/home/ema/Workspace/Projects/Kaggle/Loan_Default_Prediction/data/train_tmp.liblinear")
#            
#            if score > best_acc:
#                best_acc = score
#                best_eliminated_features = eliminated_features
#                print "Generation " + str(generation) + " - new best Acc ......" 
#                print "Mean Acc = " + str(best_acc)
#                print best_eliminated_features
#                print "\n"
#                    
#    generation += 1
#        
        
    
#    datasets.dump_svmlight_file(train, train_loss_array_libsvm, "/home/ema/Workspace/Projects/Kaggle/Loan_Default_Prediction/data/train_tmp.liblinear", zero_based=False, comment=None, query_id=None)


if __name__ == "__main__":
    
    eliminate_features()
