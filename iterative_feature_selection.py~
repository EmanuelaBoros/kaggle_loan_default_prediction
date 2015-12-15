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

from scipy.stats.stats import pearsonr
from sklearn.svm import SVR

import os.path

def select_from_sample():

    train = pandas.read_csv('data/train_v2_sample_10k.csv')
    test = pandas.read_csv('data/test_v2_sample_10k.csv')
    train_loss = train.loss
    train_loss = train_loss.apply(lambda x: 1 if x>0 else 0)
    
    column_names = train.columns.values.tolist()
    
    #train = train.drop('loss',1)
    
    ####### From the 10K sample ################"
    
    ### 3 Features best: ###
    initial_features =['f527','f528', 'f274']
    initial_auc = 0.940795486371
    
    ### 4 Features best: ###
    initial_features = ['f527', 'f528', 'f274', 'f271']
    initial_auc = 0.950707992227
    
    ### 5 Features best: ###
    initial_features = ['f527', 'f528', 'f274', 'f271', 'f334']
    initial_auc = 0.96210724728
    
    ### 6 Features best: ###
    initial_features = ['f527', 'f528', 'f274', 'f271', 'f334', 'f515']
    initial_auc = 0.971224807444
    
    ### 7 Features best: ###
    initial_features = ['f527', 'f528', 'f274', 'f271', 'f334', 'f515', 'f138']
    initial_auc = 0.971572460401
    
    ### 8 Features best: ###
    initial_features = ['f527', 'f528', 'f274', 'f271', 'f334', 'f515', 'f138', 'f617']
    initial_auc = 0.971890033905
    
    max_auc = initial_auc
    best_features = initial_features
    
    for i in range(1,778):
        f_i = "f" + str(i)
        if f_i not in initial_features and f_i in column_names:
            features = initial_features + [f_i]
    
            print i
            train_i = train[features]
            test_i = test[features]
            
            imp = Imputer()
            imp.fit(train_i)
            
            train_i = imp.transform(train_i)
            test_i = imp.transform(test_i)
            
            train_i=pre.StandardScaler().fit_transform(train_i)
            test_i=pre.StandardScaler().fit_transform(test_i)             
        
            clf = LogisticRegression(C=1e20,penalty='l2')        
           
            auc_results = cross_validation.cross_val_score(clf, train_i, train_loss, cv=5,   scoring='roc_auc')
            mean_auc = np.mean(auc_results)
            if mean_auc >  max_auc:
                max_auc = mean_auc
                best_features = features        
                print "Mean AUC = " + str(mean_auc)
                print best_features
    
    print "Final: "
    print "Mean AUC = " + str(max_auc)
    print best_features

def select_from_all_data(max_features ,train, test, train_loss, column_names):    
        
    ### 3 Features best: ###
    initial_features =['f527','f528', 'f274']
#    initial_auc = 0.942227469693
    initial_auc = 0.940795486371
    min_generation = 4
   
    max_auc_generation_dict = {}
    best_features_generation_dict = {}
    
    if os.path.isfile("data/max_auc_generation_dict.pkl"):
        print "Loading saved values "
        max_auc_generation_dict = pickle.load(open("data/max_auc_generation_dict.pkl", "rb" ) )
        best_features_generation_dict = pickle.load(open("data/best_features_generation_dict.pkl", "rb" ) )
        
        last_generation = max(max_auc_generation_dict.keys())
        min_generation =  last_generation + 1    
        initial_auc = max_auc_generation_dict[last_generation]
        initial_features = best_features_generation_dict[last_generation]
        
    else:
        print "Starting from default values "

    max_auc = initial_auc
    best_features = initial_features    
   
    
    for generation in range(min_generation, max_features + 1):
        
        print "Starting generation " + str(generation)
        print "\n"
        
        initial_features = best_features
        
        for i in range(1,779):
            f_i = "f" + str(i)
            if f_i not in initial_features and f_i in column_names:
                features = initial_features + [f_i]
        
#                print i
                train_i = train[features]
                test_i = test[features]
                
                imp = Imputer()
                imp.fit(train_i)
                
                train_i = imp.transform(train_i)
                test_i = imp.transform(test_i)
                
                train_i=pre.StandardScaler().fit_transform(train_i)
                test_i=pre.StandardScaler().fit_transform(test_i)             
            
                clf = LogisticRegression(C=1e20,penalty='l2')        
               
                auc_results = cross_validation.cross_val_score(clf, train_i, train_loss, cv=5,   scoring='roc_auc')
                mean_auc = np.mean(auc_results)
                if mean_auc >  max_auc:
                    max_auc = mean_auc
                    best_features = features        
                    print "Generation " + str(generation) + " - new best AUC ......" 
                    print "Mean AUC = " + str(mean_auc)
                    print best_features
                    print "\n"
        
        print "Generation " + str(generation) + " ...... FINAL ......" 
        print "Mean AUC = " + str(max_auc)
        print best_features
        print "\n..............................\n"
        
        max_auc_generation_dict[generation] = max_auc
        best_features_generation_dict[generation] = best_features
        
        pickle.dump( max_auc_generation_dict, open( "data/max_auc_generation_dict.pkl", "wb" ) )
        pickle.dump( best_features_generation_dict, open( "data/best_features_generation_dict.pkl", "wb" ) )
        

        
def select_from_all_data_regression(max_features ,train, test, train_loss, train_loss_array,  column_names, run):    
        
    ### 3 Features best: ###
#    initial_features =['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431']
    initial_features =['f527', 'f528']
#    initial_auc = 0.942227469693
    initial_pearson = -1.
    min_generation = 1
    
    if run == 1:
        initial_features = ['f527', 'f528']
    if run == 2:
        initial_features = ['f527',  'f274']
    if run == 3:
        initial_features = ['f527', 'f271']
    if run == 4:
        initial_features = ['f528', 'f274']
    if run == 5:
        initial_features = ['f528', 'f271']
    if run == 6:
        initial_features = ['f274', 'f271']
    if run ==7:
        initial_features = ['f404', 'f67']
   
    max_pearson_generation_dict = {}
    best_features_generation_dict = {}
    
    if os.path.isfile("data/max_pearson_generation_dict_run_" + str(run) + ".pkl"):
        print "Loading saved values "
        max_pearson_generation_dict = pickle.load(open("data/max_pearson_generation_dict_run_" + str(run) + ".pkl", "rb" ) )
        best_features_regression_generation_dict = pickle.load(open("data/best_features_regression_generation_dict_run_" + str(run) + ".pkl", "rb" ) )
        
        last_generation = max(max_pearson_generation_dict.keys())
        min_generation =  last_generation + 1    
        initial_pearson = max_pearson_generation_dict[last_generation]
        initial_features = best_features_regression_generation_dict[last_generation]
        
    else:
        print "Starting from default values "

    max_pearson = initial_pearson
    best_features = initial_features    
   
    average_best_t = 0.164463473639  
#    
#    train_binary = train.copy()
#    test_binary = test.copy()
    
#    train_binary = train_binary[['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431', 'f757']]
#    test_binary = test_binary[['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431', 'f757']]
#
##    train = train[['f527', 'f528', 'f274', 'f271']]
##    test = test[['f527', 'f528', 'f274', 'f271']]
#    
#    imp = Imputer()
#    imp.fit(train_binary)
#    
#    train_binary = imp.transform(train_binary)
#    test_binary = imp.transform(test_binary)
#    
#    train_binary=pre.StandardScaler().fit_transform(train_binary)
#    test_binary=pre.StandardScaler().fit_transform(test_binary)    
#    

    kf = pickle.load(open("data/LogisticRegression_StratifiedKFold.pkl", "rb" ) )
    
    for generation in range(min_generation, max_features + 1):
        
        print "Starting generation " + str(generation)
        print "\n"
        
        initial_features = best_features
        print initial_features
        for i_feature in range(1,779):
            f_i = "f" + str(i_feature)
            if f_i not in initial_features and f_i in column_names:     
                
                features = initial_features + [f_i]
                
#                print i
                train_i_feature = train[features]
#                test_i_feature = test[features]
                
                imp = Imputer()
                imp.fit(train_i_feature)
                
                train_i_feature = imp.transform(train_i_feature)
#                test_i_feature = imp.transform(test_i_feature)
                
                train_i_feature=pre.StandardScaler().fit_transform(train_i_feature)
#                test_i_feature=pre.StandardScaler().fit_transform(test_i_feature)   
                
                #      
                                             
#                print "Calculate MAE and Pearson with average best threshold "
                average_mae = 0.
                average_pearson = 0.
                split = 0
                for train_i, test_i in kf:
                    split += 1
            #        print len(train_i)
                    X_train_split, X_test_split, y_train_split, y_test_split = train_i_feature[train_i], train_i_feature[test_i], train_loss_array[train_i], train_loss_array[test_i]
                    y_test_split_initial = train_loss[test_i].values
                    y_train_split_initial = train_loss[train_i].values    
                    
#                    X_train_split_binary, X_test_split_binary  = train_binary[train_i], test_binary[test_i]            
                    
#                    clf = LogisticRegression(C=1e20,penalty='l2')
#                
#                    clf.fit(X_train_split_binary, y_train_split)      
#                    probas_ = clf.predict_proba(X_test_split_binary)
                    
            #        
                    probas_ = pickle.load(open("data/LogisticRegression_probas_split_" + str(split) + ".pkl", "rb" ) )
                    prediction_proba = probas_[:,1]
#                    print "... " + str(len(y_test_split_initial)) + " " + str(len(probas_))
                    
                    predictionIndexes0 = np.where(prediction_proba <= average_best_t)[0]
                    predictionIndexes1 = np.where(prediction_proba > average_best_t)[0]       
                       
#                    print "Prediction counts: .. " + str(len(predictionIndexes1)) + " " + str(len(predictionIndexes0)) + " " + str(len(predictionIndexes0) + len(predictionIndexes1)) 
                    
                    train_non_zero_indexes = np.where(y_train_split_initial > 0)[0]
  
            #        print "Train counts: " + str(len(train_non_zero_indexes)) + " " + str(len(train_zero_indexes))
                    
                    X_train_split_non_zero = X_train_split[train_non_zero_indexes]
                    y_train_split_non_zero = y_train_split_initial[train_non_zero_indexes]
                    
                    X_test_split_non_zero = X_test_split[predictionIndexes1]
                    y_test_split_initial_non_zero = y_test_split_initial[predictionIndexes1]                   
          
                    
            #        print "Real test counts: " + str(len(y_test_split_initial_non_zero_actual)) + " " + str(len(y_test_split_initial_zero_actual))
                    
                    ##############" GradientBoostingRegressor ##########################
            #        regr = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=500, 
            #                                  subsample=1.0, min_samples_split=2, min_samples_leaf=1, 
            #                                  max_depth=3, init=None, random_state=None, max_features=None, 
            #                                  alpha=0.9, verbose=0)
            #                                  
                    ### Best MAE : 0.714 with learning_rate=0.1, n_estimators=500,  predictions / 3 ###
                                              
                    ##############" LinearRegression ##########################
                    
                    regr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
                    
                    ##############" Ridge ##########################
                    
            #        regr = Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.01, solver='auto')
                    
                    ##############" LogisticRegression ##########################
                    
            #        regr = LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
            #                             C=1e20, fit_intercept=True, intercept_scaling=1.0, 
            #                             class_weight=None, random_state=None)
                                              
#                    print str(split) + "... " 
#                    print X_train_split_non_zero[0]

#                    regr = SVR(kernel='rbf', degree=2, gamma=0.0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, probability=False, cache_size=200, verbose=False, max_iter=-1, random_state=None)
               
                    regr.fit(X_train_split_non_zero, y_train_split_non_zero)
#                    
                    predicted_regr_values = regr.predict(X_test_split_non_zero)
#                    print predicted_regr_values.shape
                    
                    predicted_regr_values_norm = np.zeros(shape=(len(predicted_regr_values)))
#                    print predicted_regr_values_norm.shape
                    for i, x in enumerate(predicted_regr_values):
                        if x < 0:
                            predicted_regr_values_norm[i] = 0.
                        elif x > 100:
                            predicted_regr_values_norm[i] = 100.
                        else:
                            predicted_regr_values_norm[i] = x
                    predicted_regr_values = predicted_regr_values_norm
                            
#                    print predicted_regr_values[:10]
#                    print y_test_split_initial_non_zero[:10]       

#                    print X_test_split_non_zero[0]
                    
                    pearson = pearsonr(predicted_regr_values, y_test_split_initial_non_zero)[0]
                    average_pearson += pearson
#                    print "Pearson : " + str(pearson)
            
                    predicted_regr_values = predicted_regr_values / 3.
                        
                    prediction = np.asarray([0.] * y_test_split_initial.shape[0])
                    prediction[predictionIndexes1] = predicted_regr_values
                    prediction[predictionIndexes0] = 0.
                    mae = mean_absolute_error(y_test_split_initial, prediction)
#                    print "MAE :" + str(mae)      
                    average_mae += mae                          
                    
                              
                average_mae = average_mae / 5.
                average_pearson = average_pearson / 5.
                
                if  average_pearson >  max_pearson:
                    max_pearson = average_pearson
                    best_features = features        
                    print "Generation " + str(generation) + " - new best AUC ......" 
                    print "Mean Pearson = " + str(average_pearson)
                    print "Mean MAE: " + str(average_mae)
                    print best_features
                    print "\n"            
                    print predicted_regr_values[:10]
                    print y_test_split_initial_non_zero[:10]  
                          
            
        
        print "Generation " + str(generation) + " ...... FINAL ......" 
        print "Mean AUC = " + str(max_pearson)
        print best_features
        print "\n..............................\n"
        
        max_pearson_generation_dict[generation] = max_pearson
        best_features_generation_dict[generation] = best_features
        
        pickle.dump( max_pearson_generation_dict, open( "data/max_pearson_generation_dict_run_" + str(run) + ".pkl", "wb" ) )
        pickle.dump( best_features_generation_dict, open( "data/best_features_regression_generation_dict_run_" + str(run) + ".pkl", "wb" ) )       
        
def select_from_all_data_max_PR_area(max_features ,train, test, train_loss, train_loss_array,  column_names, run):    
        
    ### 3 Features best: ###
#    initial_features =['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431']
    initial_features =['f527', 'f528']
#    initial_auc = 0.942227469693
    initial_pr = -1.
    min_generation = 1
    
    if run == 1:
        initial_features = ['f527', 'f528']
    if run == 2:
        initial_features = ['f527',  'f274']
    if run == 3:
        initial_features = ['f527', 'f271']
    if run == 4:
        initial_features = ['f528', 'f274']
    if run == 5:
        initial_features = ['f528', 'f271']
    if run == 6:
        initial_features = ['f274', 'f271']
    if run ==7:
        initial_features = ['f527', 'f528', 'f274', 'f271', 'f334', 'f515', 'f138', 'f617']
   
    max_pr_generation_dict = {}
    best_features_pr_generation_dict = {}
    
    if os.path.isfile("data/max_pr_generation_dict_run_" + str(run) + ".pkl"):
        print "Loading saved values "
        max_pr_generation_dict = pickle.load(open("data/max_pr_generation_dict_run_" + str(run) + ".pkl", "rb" ) )
        best_features_pr_generation_dict = pickle.load(open("data/best_features_pr_generation_dict_run_" + str(run) + ".pkl", "rb" ) )
        
        last_generation = max(max_pr_generation_dict.keys())
        min_generation =  last_generation + 1    
        initial_pr = max_pr_generation_dict[last_generation]
        initial_features = best_features_pr_generation_dict[last_generation]
        
    else:
        print "Starting from default values "

    max_pr = initial_pr
    best_features = initial_features    
   
#    average_best_t = 0.164463473639      


    kf = pickle.load(open("data/LogisticRegression_StratifiedKFold.pkl", "rb" ) )
    
    for generation in range(min_generation, max_features + 1):
        
        print "Starting generation " + str(generation)
        print "\n"
        
        initial_features = best_features
        print initial_features
        for i_feature in range(1,779):
            f_i = "f" + str(i_feature)
            if f_i not in initial_features and f_i in column_names:     
                
                features = initial_features + [f_i]
                
#                print i
                train_i_feature = train[features]
#                test_i_feature = test[features]
                
                imp = Imputer()
                imp.fit(train_i_feature)
                
                train_i_feature = imp.transform(train_i_feature)
#                test_i_feature = imp.transform(test_i_feature)
                
                train_i_feature=pre.StandardScaler().fit_transform(train_i_feature)
#                test_i_feature=pre.StandardScaler().fit_transform(test_i_feature)   
                
                #      
                                             
#                print "Calculate MAE and Pearson with average best threshold "
                average_pr = 0.                
                split = 0
                for train_i, test_i in kf:
                    split += 1
            #        print len(train_i)
                    X_train_split, X_test_split, y_train_split, y_test_split = train_i_feature[train_i], train_i_feature[test_i], train_loss_array[train_i], train_loss_array[test_i]
                    y_test_split_initial = train_loss[test_i].values
#                    y_train_split_initial = train_loss[train_i].values    
                    
#                                         
                    clf = SGDClassifier(loss='log', penalty='l2', alpha=1e20, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.1, n_jobs=6, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, rho=None, seed=None)
           
                    clf.fit(X_train_split,y_train_split)      
                    probas_ = clf.predict_proba(X_test_split)
#                    prediction = clf.predict(X_test_split)
                                        
                    precision, recall, thresholds = metrics.precision_recall_curve(y_test_split, probas_[:, 1])
                    pr_area_split = auc(recall, precision)
                    average_pr += pr_area_split     
                    
                              
                average_pr = average_pr / 5.
              
                
                if  average_pr >  max_pr:     
                    max_pr = average_pr
                    best_features = features        
                    print "Generation " + str(generation) + " - new best PR Area ......" 
                    print "Mean PR Area = " + str(average_pr)                 
                    print best_features
                    print "\n"            
#                    print predicted_regr_values[:10]
#                    print y_test_split_initial_non_zero[:10]                            
            
        
        print "Generation " + str(generation) + " ...... FINAL ......" 
        print "Mean PR Area = " + str(max_pr)
        print best_features
        print "\n..............................\n"
        
        max_pr_generation_dict[generation] = max_pr
        best_features_pr_generation_dict[generation] = best_features
        
        pickle.dump( max_pr_generation_dict, open( "data/max_pr_generation_dict_run_" + str(run) + ".pkl", "wb" ) )
        pickle.dump( best_features_pr_generation_dict, open( "data/best_features_pr_generation_dict_run_" + str(run) + ".pkl", "wb" ) )       

if __name__ == "__main__":
#    select_from_sample()

#    train = pandas.read_csv('data/train_v2.csv')
#    test = pandas.read_csv('data/test_v2.csv')
#    train_loss = train.loss
#    train_loss = train_loss.apply(lambda x: 1 if x>0 else 0)
#    column_names = train.columns.values.tolist()
#    
#    max_features = 50    
#    select_from_all_data(max_features,train, test, train_loss, column_names)
    
    train = pandas.read_csv('data/train_v2.csv')
#    test = pandas.read_csv('data/test_v2.csv')
    test = ""
    train_loss = train.loss.copy()   
    train_loss_array = train_loss.apply(lambda x: 1 if x>0 else 0).values
    column_names = train.columns.values.tolist()
    
    max_features = 50    
    run = 7
    select_from_all_data_max_PR_area(max_features,train, test, train_loss,train_loss_array, column_names, run)
