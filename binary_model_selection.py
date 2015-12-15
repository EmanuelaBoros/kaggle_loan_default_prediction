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

from sklearn.metrics import mean_absolute_error, accuracy_score

from scipy.stats.stats import pearsonr, spearmanr
from sklearn.svm import SVR

import os.path

from sklearn.naive_bayes import GaussianNB

def select_binary_model():

    use_sample = True
    
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

    train_full = train.copy()
#    test_full = test.copy()   
    
#    cols = set(train_full.columns)
#    cols.remove('loss')
#    cols = list(cols)
#    train_full = train_full[cols]
    
#    train_full = train_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']]
    train_full = train_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f292', 'f73', 'f419', 'f277', 'f298']]
   
#    train_full = train_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f292', 'f73', 'f419', 'f277', 'f298', 'f377', 'f112', 'f746', 'f135', 'f428', 'f667', 'f336', 'f272', 'f239', 'f27', 'f499', 'f775', 'f233', 'f334', 'f337', 'f418', 'f241', 'f335', 'f338', 'f648', 'f81', 'f471', 'f518', 'f679', 'f635', 'f16', 'f69', 'f617', 'f148', 'f164', 'f588', 'f25', 'f765', 'f556', 'f83']]

#    test_full = test_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']]
        
    ## Best Spearman ##
#    train_full = train_full[['f527', 'f274', 'f776', 'f1', 'f619', 'f5', 'f650', 'f216']]
    
    imp = Imputer()
    imp.fit(train_full)
    
    train_full = imp.transform(train_full)
#    test_full = imp.transform(test_full)
    
    train_full=pre.StandardScaler().fit_transform(train_full)
#    test_full=pre.StandardScaler().fit_transform(test_full) 
    
    
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
    
    kf = StratifiedKFold(train_loss.values, n_folds=5, indices=False)
    train_loss_array = train_loss.apply(lambda x: 1 if x>0 else 0).values     
    
#    print train_loss_array.shape

    
#    print "Average best threshold " + str(average_best_t)
    
#    print "Calculate MAE with average best threshold "
    average_mae = 0.
    average_f1 = 0.
    average_auc = 0.
    average_pr_area = 0.
    
    regr_predicted_values_full = []
    regr_X_full = []
    regr_y_full = []
#    print regr_predicted_values_full.shape

    
#    clf = SGDClassifier(loss='log', penalty='l2', alpha=1e-10, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.1, n_jobs=6, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, rho=None, seed=None)
       
    
#    clf.fit(train,train_loss_array)      
#    train = clf.transform(train, threshold = "3*mean")
#    print train.shape

    train_loss_array_libsvm = train_loss.apply(lambda x: 1 if x>0 else -1).values
    datasets.dump_svmlight_file(train, train_loss_array_libsvm, "/home/ema/Workspace/Projects/Kaggle/Loan_Default_Prediction/data/train_full_selected.liblinear", zero_based=False, comment=None, query_id=None)
    
    for train_i, test_i in kf:
#        print len(train_i)
        X_train_split, X_test_split, y_train_split, y_test_split = train[train_i], train[test_i], train_loss_array_libsvm[train_i], train_loss_array_libsvm[test_i]
        y_test_split_initial = train_loss[test_i].values
        y_train_split_initial = train_loss[train_i].values
        
        X_train_full_split, X_test_full_split = train_full[train_i], train_full[test_i]        

########## Predict Log Reg ################        
        clf = LogisticRegression(C=1e20,penalty='l2')
        
        clf = GaussianNB()
        
        clf = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.1, n_jobs=6, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, rho=None, seed=None)
       
        clf = LinearSVC(penalty='l2', loss='l2', dual=False, tol=0.0001, C=0.001, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight="auto", verbose=0, random_state=None)
    
        print "Train classif: "
        clf.fit(X_train_split,y_train_split)      
#        probas_ = clf.predict_proba(X_test_split)
        prediction = clf.predict(X_test_split)
        print prediction
        
#        print "... " + str(len(y_test_split_initial)) + " " + str(len(probas_))
        
#        prediction_proba = probas_[:,1]

#        f1_split = 0.
#        for i in np.arange(0.1,0.2,0.01):       
#        
#            predictionIndexes0 = np.where(prediction_proba <= i)[0]
#            predictionIndexes1 = np.where(prediction_proba > i)[0]   
#            prediction_threshold = np.asarray([0.] * y_test_split_initial.shape[0])
#            prediction_threshold[predictionIndexes1] = 1.
#            prediction_threshold[predictionIndexes0] = 0.
#            
#            f1_split_t = metrics.f1_score(y_test_split, prediction_threshold, average='macro')
#            if f1_split_t > f1_split:
#                f1_split = f1_split_t
#                average_best_t = i
            
        
        
#        print("Area under the ROC curve : %f" % roc_auc)

        predictionIndexes0 = np.where(prediction == 0)[0]
        predictionIndexes1 = np.where(prediction == 1)[0]   
        prediction_threshold = np.asarray([0.] * y_test_split_initial.shape[0])
        prediction_threshold[predictionIndexes1] = 1.
        prediction_threshold[predictionIndexes0] = 0
        
        print len(predictionIndexes1)
        
        print prediction
        
        f1_split = metrics.f1_score(y_test_split, prediction)
        average_f1 += f1_split
        
#        fpr, tpr, thresholds = roc_curve(y_test_split, probas_[:, 1])
#        average_auc += auc(fpr, tpr)
        
#        precision, recall, thresholds = metrics.precision_recall_curve(y_test_split, probas_[:, 1])
#        pr_area_split = auc(recall, precision)
#        average_pr_area += pr_area_split
        

        prediction = np.asarray([0.] * y_test_split_initial.shape[0])
        prediction[predictionIndexes1] = 1
        prediction[predictionIndexes0] = 0.
        mae = mean_absolute_error(y_test_split_initial, prediction)

        default_value = 4.5
        for i in np.arange(0.1,20.,0.1):
            prediction[predictionIndexes1] = i
            mae_value = mean_absolute_error(y_test_split_initial, prediction)
            if mae_value < mae:
                 mae = mae_value
                 default_value = i
        
        
        print "MAE :" + str(mae) + " value: " + str(default_value) + " threshold: "  + str(average_best_t)    
        average_mae += mae          

#        pl.scatter(predicted_regr_values, y_test_split_initial_non_zero, c='k', label='data')       
#        pl.show()         
#        
                  
    average_mae = average_mae / 5.
    average_f1 = average_f1 / 5.
    average_auc =average_auc / 5.
    average_pr_area = average_pr_area / 5.
    print "Mean MAE: " + str(average_mae)   
    print "Mean F1: " + str(average_f1) 
    print "Mean AUC: " + str(average_auc) 
    print "Mean PR Area: " + str(average_pr_area)
    
#    regr_predicted_values_full = np.array(regr_predicted_values_full)
#    regr_X_full = np.array(regr_X_full)
#    regr_y_full = np.array(regr_y_full)
#    
#    print regr_predicted_values_full.shape )       

def select_best_svm():

    use_sample = True
    
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
   
    train = train[['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431', 'f757']]
#    test = test[['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431', 'f757']]

#    train = train[['f527', 'f528', 'f274', 'f271']]
#    test = test[['f527', 'f528', 'f274', 'f271']]
    
    imp = Imputer()
    imp.fit(train)
    
    train = imp.transform(train)
#    test = imp.transform(test)
    
    train=pre.StandardScaler().fit_transform(train)
#    test=pre.StandardScaler().fit_transform(test)    
    
    kf = StratifiedKFold(train_loss.values, n_folds=5, indices=False)
    train_loss_array = train_loss.apply(lambda x: 1 if x>0 else 0).values     
#    print train_loss_array.shape

    
#    print "Average best threshold " + str(average_best_t)
    
#    print "Calculate MAE with average best threshold "
  
    
#    regr_predicted_values_full = []
#    regr_X_full = []
#    regr_y_full = []
#    print regr_predicted_values_full.shape

    best_pr_area = 0.
    best_c = 0.
    kernel_type = "linear"
    C_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1 , 1e1, 1e2 ,1e3 , 1e4, 1e5]
    for c_value in C_values:
        average_mae = 0.
        average_f1 = 0.
        average_auc = 0.
        average_pr_area = 0.
        clf = svm.SVC(C=c_value, kernel=kernel_type, degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
        print c_value
        for train_i, test_i in kf:
    #        print len(train_i)
            X_train_split, X_test_split, y_train_split, y_test_split = train[train_i], train[test_i], train_loss_array[train_i], train_loss_array[test_i]
            y_test_split_initial = train_loss[test_i].values
    #        y_train_split_initial = train_loss[train_i].values
    #        
    #        X_train_full_split, X_test_full_split = train_full[train_i], train_full[test_i]        
    
    ########## Predict Log Reg ################                     
           
        
            clf.fit(X_train_split,y_train_split)      
            probas_ = clf.predict_proba(X_test_split)
#            prediction = clf.predict(X_test_split)
            
    #        print "... " + str(len(y_test_split_initial)) + " " + str(len(probas_))
            
#            prediction_proba = probas_[:,1]
    
    #        f1_split = 0.
    #        for i in np.arange(0.1,0.2,0.01):       
    #        
    #            predictionIndexes0 = np.where(prediction_proba <= i)[0]
    #            predictionIndexes1 = np.where(prediction_proba > i)[0]   
    #            prediction_threshold = np.asarray([0.] * y_test_split_initial.shape[0])
    #            prediction_threshold[predictionIndexes1] = 1.
    #            prediction_threshold[predictionIndexes0] = 0.
    #            
    #            f1_split_t = metrics.f1_score(y_test_split, prediction_threshold, average='macro')
    #            if f1_split_t > f1_split:
    #                f1_split = f1_split_t
    #                average_best_t = i
                
            
            
    #        print("Area under the ROC curve : %f" % roc_auc)
    
#            predictionIndexes0 = np.where(prediction_proba <= average_best_t)[0]
#            predictionIndexes1 = np.where(prediction_proba > average_best_t)[0]   
#            prediction_threshold = np.asarray([0.] * y_test_split_initial.shape[0])
#            prediction_threshold[predictionIndexes1] = 1.
#            prediction_threshold[predictionIndexes0] = 0
            
#            f1_split = metrics.f1_score(y_test_split, prediction_threshold, average='macro')
#            average_f1 += f1_split
            
#            fpr, tpr, thresholds = roc_curve(y_test_split, probas_[:, 1])
#            average_auc += auc(fpr, tpr)
            
            precision, recall, thresholds = metrics.precision_recall_curve(y_test_split, probas_[:, 1])
            pr_area_split = auc(recall, precision)
            average_pr_area += pr_area_split
            
                
#            prediction = np.asarray([0.] * y_test_split_initial.shape[0])
#            prediction[predictionIndexes1] = 1.
#            prediction[predictionIndexes0] = 0.
#            mae = mean_absolute_error(y_test_split_initial, prediction)
    
            default_value = 1.
    #        for i in np.arange(0.1,20.,0.1):
    #            prediction[predictionIndexes1] = i
    #            mae_value = mean_absolute_error(y_test_split_initial, prediction)
    #            if mae_value < mae:
    #                 mae = mae_value
    #                 default_value = i
            
            
#            print "MAE :" + str(mae) + " value: " + str(default_value) + " threshold: "  + str(average_best_t)    
#            average_mae += mae          
    
    #        pl.scatter(predicted_regr_values, y_test_split_initial_non_zero, c='k', label='data')       
    #        pl.show()         
    #        
                      
#        average_mae = average_mae / 5.
#        average_f1 = average_f1 / 5.
#        average_auc =average_auc / 5.
        average_pr_area = average_pr_area / 5.
#        print "Mean MAE: " + str(average_mae)   
#        print "Mean F1: " + str(average_f1) 
#        print "Mean AUC: " + str(average_auc) 
        print "Mean PR Area: " + str(average_pr_area)
        
        if average_pr_area > best_pr_area:
            best_pr_area = average_pr_area
            best_c = c_value
    
    print "Best C: " + str(best_c) + " Mean PR Area: " + str(best_pr_area)

if __name__ == "__main__":
    
    select_binary_model()
