import pandas

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn import cross_validation

from sklearn.preprocessing import Imputer

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing as pre
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm, datasets
from sklearn.metrics import mean_absolute_error

import statsmodels.api as sm

import numpy as np
import pylab as pl
from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn import metrics
import pickle
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier

import os.path

def find_threshold():

    train = pandas.read_csv('data/train_v2.csv')
    test = pandas.read_csv('data/test_v2.csv')
    train_loss = train.loss
       
    train = train[['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431', 'f757']]
    test = test[['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431', 'f757']]
    
#    train = train[['f527', 'f528', 'f274', 'f271']]
#    test = test[['f527', 'f528', 'f274', 'f271']]
    
    imp = Imputer()
    imp.fit(train)
    
    train = imp.transform(train)
    test = imp.transform(test)
    
    train=pre.StandardScaler().fit_transform(train)
    test=pre.StandardScaler().fit_transform(test)
    
    
    kf = StratifiedKFold(train_loss.values, n_folds=5, indices=False)
    train_loss_array = train_loss.apply(lambda x: 1 if x>0 else 0).values

    mean_auc = 0.0    
    average_best_t = 0.
    for train_i, test_i in kf:
#        print len(train_i)
        X_train_split, X_test_split, y_train_split, y_test_split = train[train_i], train[test_i], train_loss_array[train_i], train_loss_array[test_i]
        y_test_split_initial = train_loss[test_i].values
        
        clf = LogisticRegression(C=1e20,penalty='l2')
    
        clf.fit(X_train_split,y_train_split)      
        probas_ = clf.predict_proba(X_test_split)
        prediction_proba = probas_[:,1]
        
        fpr, tpr, thresholds = roc_curve(y_test_split, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        print roc_auc
        mean_auc += roc_auc
        
        min_mae = 1.
        best_t = 0.
#        print len(thresholds)
        for i, t in enumerate(thresholds):        
            
            predictionIndexes0 = np.where(prediction_proba <= t)[0]
            predictionIndexes1 = np.where(prediction_proba > t)[0]
            
            prediction = np.asarray([0.] * y_test_split_initial.shape[0])
            prediction[predictionIndexes1] = 1.
            prediction[predictionIndexes0] = 0.
                    
            mae = mean_absolute_error(y_test_split_initial, prediction) 
            if mae < min_mae:
                min_mae = mae
                best_t = t
#                print str(min_mae) + " " + str(best_t)
                
        print str(min_mae) + " " + str(best_t)
        average_best_t += best_t

    mean_auc = mean_auc / 5.
    print "Mean AUC: " + str(mean_auc)
    
    average_best_t = average_best_t / 5.
    
    print "Average best threshold " + str(average_best_t)
    
    print "Calculate MAE with average best threshold "
    average_mae = 0.
    for train_i, test_i in kf:
#        print len(train_i)
        X_train_split, X_test_split, y_train_split, y_test_split = train[train_i], train[test_i], train_loss_array[train_i], train_loss_array[test_i]
        y_test_split_initial = train_loss[test_i].values
        
        clf = LogisticRegression(C=1e20,penalty='l2')
    
        clf.fit(X_train_split,y_train_split)      
        probas_ = clf.predict_proba(X_test_split)
        prediction_proba = probas_[:,1]
        
        prediction_proba = clf.predict(X_test_split)
        
        predictionIndexes0 = np.where(prediction_proba <= average_best_t)[0]
        predictionIndexes1 = np.where(prediction_proba > average_best_t)[0]
            
        prediction = np.asarray([0.] * y_test_split_initial.shape[0])
        prediction[predictionIndexes1] = 1.
        prediction[predictionIndexes0] = 0.
        mae = mean_absolute_error(y_test_split_initial, prediction)
        print "MAE :" + str(mae)      
        average_mae += mae    
    average_mae = average_mae / 5.
    print "Mean MAE: " + str(average_mae) 
          
    
    
#    X_train,X_test,y_train_binary,y_test_binary = train_test_split( train, train_loss.apply(lambda x: 1 if x>0 else 0), test_size=0.2, random_state=42)
#    X_train,X_test,y_train,y_test = train_test_split( train, train_loss, test_size=0.2, random_state=42)
#  
#    print y_train_binary[:10]
#    clf = LogisticRegression(C=1e20,penalty='l2')
#    
#    clf.fit(X_train,y_train_binary)
#    
#    print roc_auc_score(y_test_binary,clf.predict_proba(X_test)[:,1])
#    
#    probas_ = clf.predict_proba(X_test)
#    fpr, tpr, thresholds = roc_curve(y_test_binary, probas_[:, 1])
#    roc_auc = auc(fpr, tpr)
#    print("Area under the ROC curve : %f" % roc_auc)
#    
#    min_mae = 1.
#    best_t = 0.
#    print len(thresholds)
#    for i, t in enumerate(thresholds):        
#        
#        predictionIndexes0 = np.where(probas_ <= t)[0]
#        predictionIndexes1 = np.where(probas_ > t)[0]
#        
#        prediction = np.asarray([0.] * y_test.shape[0])
#        prediction[predictionIndexes1] = 1.
#        prediction[predictionIndexes0] = 0.
#                
#        mae = mean_absolute_error(y_test, prediction) 
#        if mae < min_mae:
#            min_mae = mae
#            best_t = t
#            print str(min_mae) + " " + str(best_t)
#            
#    print str(min_mae) + " " + str(best_t)

def find_threshold_SGD():

    train = pandas.read_csv('data/train_v2.csv')
#    test = pandas.read_csv('data/test_v2.csv')
    train_loss = train.loss
       
#    train = train[['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431', 'f757']]
#    test = test[['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431', 'f757']]
    
#    train = train[['f527', 'f528', 'f274', 'f271']]
#    test = test[['f527', 'f528', 'f274', 'f271']]
    
    imp = Imputer()
    imp.fit(train)
    
    train = imp.transform(train)
#    test = imp.transform(test)
    
    train=pre.StandardScaler().fit_transform(train)
#    test=pre.StandardScaler().fit_transform(test)
    
    
    train_loss_array = train_loss.apply(lambda x: 1 if x>0 else 0).values
    
    clf = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.1, n_jobs=6, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, rho=None, seed=None)
       
    
    clf.fit(train,train_loss_array)      
    train = clf.transform(train, threshold = "1.25*mean")
    print train.shape    
    
    kf = StratifiedKFold(train_loss.values, n_folds=5, indices=False)    

    average_f1 = 0.0    
    average_best_t = 0.
    for train_i, test_i in kf:
#        print len(train_i)
        X_train_split, X_test_split, y_train_split, y_test_split = train[train_i], train[test_i], train_loss_array[train_i], train_loss_array[test_i]
        y_test_split_initial = train_loss[test_i].values
        
        clf = SGDClassifier(loss='log', penalty='l2', alpha=1e-4, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.1, n_jobs=6, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, rho=None, seed=None)
    
        clf.fit(X_train_split,y_train_split)      
        probas_ = clf.predict_proba(X_test_split)
        prediction_proba = probas_[:,1]
        
        fpr, tpr, thresholds = roc_curve(y_test_split, probas_[:, 1])
        
        predictionIndexes0 = np.where(prediction_proba <= average_best_t)[0]
        predictionIndexes1 = np.where(prediction_proba > average_best_t)[0]   
        prediction_threshold = np.asarray([0.] * y_test_split_initial.shape[0])
        prediction_threshold[predictionIndexes1] = 1.
        prediction_threshold[predictionIndexes0] = 0
        f1_split = metrics.f1_score(y_test_split, prediction_threshold, average='macro')
        average_f1 += f1_split
        
        min_mae = 1.
        best_t = 0.
#        thresholds = np.arange(0.,1.,0.001)
        print len(thresholds)
        for index, t in enumerate(thresholds[300:400]):   
            
            if index % 1000 == 0:
                print index
            
            predictionIndexes0 = np.where(prediction_proba <= t)[0]
            predictionIndexes1 = np.where(prediction_proba > t)[0]
            
            prediction = np.asarray([0.] * y_test_split_initial.shape[0])
            prediction[predictionIndexes1] = 1.
            prediction[predictionIndexes0] = 0.
            mae = mean_absolute_error(y_test_split_initial, prediction)
            
            default_value = 1
            for i in np.arange(0.1,20.,0.1):
#                print i
                prediction[predictionIndexes1] = i
                mae_value = mean_absolute_error(y_test_split_initial, prediction)
                if mae_value < mae:
                     mae = mae_value
                     default_value = i
                    
#            mae = mean_absolute_error(y_test_split_initial, prediction) 
            if mae < min_mae:
                min_mae = mae
                best_t = t
                print str(min_mae) + " " + str(best_t) + " " + str(index) + " value: " + str(default_value)
                
        print "SPLIT " +  str(min_mae) + " " + str(best_t)
        average_best_t += best_t

    average_f1 = average_f1 / 5.
    print "Mean F1: " + str(average_f1)
    
    average_best_t = average_best_t / 5.
    
    average_best_t = 0.999999999164
    print "Average best threshold " + str(average_best_t)
    
    print "Calculate MAE with average best threshold "
    average_mae = 0.
#    for train_i, test_i in kf:
##        print len(train_i)
#        X_train_split, X_test_split, y_train_split, y_test_split = train[train_i], train[test_i], train_loss_array[train_i], train_loss_array[test_i]
#        y_test_split_initial = train_loss[test_i].values
#        
#        clf = LogisticRegression(C=1e20,penalty='l2')
#        ### MAE 0.55 With LogReg
#        clf = SGDClassifier(loss='log', penalty='l2', alpha=1e-4, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.1, n_jobs=6, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, rho=None, seed=None)
#    
#        clf.fit(X_train_split,y_train_split)      
#        probas_ = clf.predict_proba(X_test_split)
#        prediction_proba = probas_[:,1]
#        
#        prediction_proba = clf.predict(X_test_split)
#        
#        predictionIndexes0 = np.where(prediction_proba <= average_best_t)[0]
#        predictionIndexes1 = np.where(prediction_proba > average_best_t)[0]   
#        prediction_threshold = np.asarray([0.] * y_test_split_initial.shape[0])
#        prediction_threshold[predictionIndexes1] = 1.
#        prediction_threshold[predictionIndexes0] = 0
#        f1_split = metrics.f1_score(y_test_split, prediction_threshold, average='macro')
#        average_f1 += f1_split
#            
#        prediction = np.asarray([0.] * y_test_split_initial.shape[0])
#        prediction[predictionIndexes1] = 1.
#        prediction[predictionIndexes0] = 0.
#        mae = mean_absolute_error(y_test_split_initial, prediction)
#        
#        default_value = 1.
#        for i in np.arange(0.1,20.,0.1):
#            prediction[predictionIndexes1] = i
#            mae_value = mean_absolute_error(y_test_split_initial, prediction)
#            if mae_value < mae:
#                 mae = mae_value
#                 default_value = i

    for train_i, test_i in kf:
#        print len(train_i)
        X_train_split, X_test_split, y_train_split, y_test_split = train[train_i], train[test_i], train_loss_array[train_i], train_loss_array[test_i]
        y_test_split_initial = train_loss[test_i].values
        
        clf = SGDClassifier(loss='log', penalty='l2', alpha=1e-4, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.1, n_jobs=6, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, rho=None, seed=None)
    
        clf.fit(X_train_split,y_train_split)      
        probas_ = clf.predict_proba(X_test_split)
        prediction_proba = probas_[:,1]
        
        fpr, tpr, thresholds = roc_curve(y_test_split, probas_[:, 1])
        
        predictionIndexes0 = np.where(prediction_proba <= average_best_t)[0]
        predictionIndexes1 = np.where(prediction_proba > average_best_t)[0]   
        prediction_threshold = np.asarray([0.] * y_test_split_initial.shape[0])
        prediction_threshold[predictionIndexes1] = 1.
        prediction_threshold[predictionIndexes0] = 0
        f1_split = metrics.f1_score(y_test_split, prediction_threshold, average='macro')
        average_f1 += f1_split
        
        min_mae = 1.
        best_t = 0.
#        thresholds = np.arange(0.,1.,0.001)
        print len(thresholds)
        thresholds = [average_best_t]
        for index, t in enumerate(thresholds):   
            
            
            if index % 1000 == 0:
                print index
            
            predictionIndexes0 = np.where(prediction_proba <= t)[0]
            predictionIndexes1 = np.where(prediction_proba > t)[0]
            
            prediction = np.asarray([0.] * y_test_split_initial.shape[0])
            prediction[predictionIndexes1] = 1.
            prediction[predictionIndexes0] = 0.
            mae = mean_absolute_error(y_test_split_initial, prediction)
            
            default_value = 1
            for i in np.arange(0.1,20.,0.1):
#                print i
                prediction[predictionIndexes1] = i
                mae_value = mean_absolute_error(y_test_split_initial, prediction)
                if mae_value < mae:
                     mae = mae_value
                     default_value = i
                    
#            mae = mean_absolute_error(y_test_split_initial, prediction) 
            if mae < min_mae:
                min_mae = mae
                best_t = t
                print str(min_mae) + " " + str(best_t) + " " + str(index) + " value: " + str(default_value)
                
        print "SPLIT " +  str(min_mae) + " " + str(best_t)
        average_best_t += best_t
                     
        print "MAE :" + str(mae)   + " Value: " + str(default_value)   
        average_mae += mae    
    average_mae = average_mae / 5.
    print "Mean MAE: " + str(average_mae) 
    
def test_threshold_SGD():
    train = pandas.read_csv('data/train_v2.csv')
#    test = pandas.read_csv('data/test_v2.csv')
    train_loss = train.loss
       
#    train = train[['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431', 'f757']]
#    test = test[['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431', 'f757']]
    
#    train = train[['f527', 'f528', 'f274', 'f271']]
#    test = test[['f527', 'f528', 'f274', 'f271']]
    
    imp = Imputer()
    imp.fit(train)
    
    train = imp.transform(train)
#    test = imp.transform(test)
    
    train=pre.StandardScaler().fit_transform(train)
#    test=pre.StandardScaler().fit_transform(test)
    
    
    train_loss_array = train_loss.apply(lambda x: 1 if x>0 else 0).values
    
    clf = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.1, n_jobs=6, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, rho=None, seed=None)
       
    
    clf.fit(train,train_loss_array)      
    train = clf.transform(train, threshold = "1.25*mean")
    print train.shape    
    
    kf = StratifiedKFold(train_loss.values, n_folds=10, indices=False)    

    threshold  = 0.999999999164       
    mean_mae = 0.
    for train_i, test_i in kf:
#        print len(train_i)
        X_train_split, X_test_split, y_train_split, y_test_split = train[train_i], train[test_i], train_loss_array[train_i], train_loss_array[test_i]
        y_test_split_initial = train_loss[test_i].values
        
        clf = SGDClassifier(loss='log', penalty='l2', alpha=1e-4, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.1, n_jobs=6, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, rho=None, seed=None)
    
        clf.fit(X_train_split,y_train_split)      
        probas_ = clf.predict_proba(X_test_split)
        prediction_proba = probas_[:,1]
        
        predictionIndexes0 = np.where(prediction_proba <= threshold)[0]
        predictionIndexes1 = np.where(prediction_proba > threshold)[0]
        
        prediction = np.asarray([0.] * y_test_split_initial.shape[0])
        prediction[predictionIndexes1] = 10.
        prediction[predictionIndexes0] = 0.
        mae = mean_absolute_error(y_test_split_initial, prediction)
    
        mean_mae += mae
        
        print "Split MAE: " + str(mae)
    mean_mae = mean_mae / 10.
    print "Average MAE: " + str(mean_mae)
    
    
if __name__ == "__main__":
#    select_from_sample()

#    train = pandas.read_csv('data/train_v2.csv')
#    test = pandas.read_csv('data/test_v2.csv')
#    train_loss = train.loss
#    train_loss = train_loss.apply(lambda x: 1 if x>0 else 0)
#    column_names = train.columns.values.tolist()
    
#    max_features = 50    
#    find_threshold()
    
#    find_threshold_SGD()

     test_threshold_SGD()
