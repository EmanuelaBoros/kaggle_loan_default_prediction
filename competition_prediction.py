import pandas

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn import cross_validation

from sklearn.preprocessing import Imputer

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, SGDClassifier
from sklearn import preprocessing as pre
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor
from sklearn import svm, datasets
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import f_regression
from sklearn.svm import SVR, SVC
import pylab as pl


import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics, linear_model
import pickle

from sklearn.feature_selection import SelectKBest

from scipy.stats.stats import pearsonr, spearmanr

from glmsklearn import BinomialRegressor, GammaRegressor, GaussianRegressor, \
    InverseGaussianRegressor, NegativeBinomialRegressor, PoissonRegressor
from statsmodels.genmod.families.family import Binomial, Gamma, Gaussian,\
    InverseGaussian, NegativeBinomial, Poisson

import statsmodels

import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from scipy.linalg import inv
from statsmodels.tools.tools import chain_dot as dot


def quantilereg(y,x,p):
    '''
    estimates quantile regression based on weighted least squares.
    constant term is added to x matrix
   __________________________________________________________________________
 
    Inputs:
     y: Dependent variable
     x: matrix of independent variables
     p: quantile
 
    Outputs:
     beta:     estimated Coefficients.
     tstats:   T- students of the coefficients.
     VCboot:   Variance-Covariance of coefficients by bootstrapping method.
     itrat:    number of iterations for convergence to roots.
     PseudoR2: in quatile regression another definition of R2 is used namely PseudoR2.
     betaboot: estimated coefficients by bootstrapping method.
 
    This code can be used for quantile regression estimation as whole,and LAD
    regression as special case of it, when one sets p=0.5.
 
    Copyright(c) Shapour Mohammadi, University of Tehran, 2008
    shmohammadi@gmail.com
 
    Translated to python with permission from original author by Christian Prinoth (christian at prinoth dot name)
 
   Ref:
    1-Birkes, D. and Y. Dodge(1993). Alternative Methods of Regression, John Wiley and Sons.
    2-Green,W. H. (2008). Econometric Analysis. Sixth Edition. International Student Edition.
    3-LeSage, J. P.(1999),Applied Econometrics Using MATLAB,
 
    Keywords: Least Absolute Deviation(LAD) Regression, Quantile Regression,
    Regression, Robust Estimation.
   __________________________________________________________________________
   '''
 
    tstats=0
    VCboot=0
    itrat=0
    PseudoR2=0
    betaboot=0
 
    ry=len(y)
    rx, cx=x.shape
    x=np.c_[np.ones(rx),x]
    cx=cx+1
    #______________Finding first estimates by solving the system_______________
    # Some lines of this section is based on a code written by
    # James P. Lesage in Applied Econometrics Using MATLAB(1999).PP. 73-4.
    itrat=0
    xstar=x
    diff=1
    beta=np.ones(cx)
    z=np.zeros((rx,cx))
    while itrat<1000 and diff>1e-6:
        itrat+=1
        beta0=beta
        beta=dot(inv(dot(xstar.T,x)),xstar.T,y)
        resid=y-dot(x,beta)
        resid[np.abs(resid)<.000001]=.000001
        resid[resid<0]=p*resid[resid<0]
        resid[resid>0]=(1-p)*resid[resid>0]
        resid=np.abs(resid)
        for i in range(cx):
            z[:,i] = x[:,i]/resid
 
        xstar=z
        beta1=beta
        diff=np.max(np.abs(beta1-beta0))
 
    return beta

def predict_quantile_regr(y_train_split_non_zero, X_train_full_split_non_zero, X_test_full_split_non_zero, quantile):
    beta = quantilereg(y_train_split_non_zero,X_train_full_split_non_zero,quantile)
#    print beta
    predicted_regr_values = []
    for x_t in X_test_full_split_non_zero:
#            print len(beta)
        pred = np.dot(x_t, beta[1:]) + beta[0]
#            print pred
        
        predicted_regr_values += [pred]
                              
#    print predicted_regr_values
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
    return predicted_regr_values
    
    
def write_prediction_file(prediction_values_file, sample_file, competition_format_file):

    f_prediction_values = open(prediction_values_file, "r")
    prediction_values = []
    for line in f_prediction_values:
        prediction_values += [line.strip()]
        
    f_prediction_sample = open(sample_file, "r")
    prediction_ids = []
    f_prediction_sample.readline()
    for line in f_prediction_sample:
        prediction_ids += [line.strip().split(",")[0]]
        
    f_out = open(competition_format_file, "w")
    f_out.write("id,loss\n")
    for i, instace_id in enumerate(prediction_ids):        
        f_out.write(instace_id + "," + prediction_values[i] + "\n")
    f_out.close()
              
 
def make_prediction_1():

    use_sample = False
    
    if use_sample:
        train = pandas.read_csv('data/train_v2_sample_10k.csv')
        test = pandas.read_csv('data/test_v2_sample_10k.csv')
        average_best_t = 0.148846575958
    else:
        train = pandas.read_csv('data/train_v2.csv')
        test = pandas.read_csv('data/test_v2.csv')
         ### To use on full train set
        average_best_t = 0.164463473639
  
    
    train_loss = train.loss
    train_full = train.copy()
    test_full = test.copy()   
    
    binary_classification_feature_set_1 = ['f527', 'f528', 'f274', 'f271']
    binary_classification_feature_set_2 = ['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431', 'f757']
    
    regression_feature_set_1 = ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']
    regression_feature_set_2 = ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']
    regression_feature_set_3 = ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']
    regression_feature_set_4 = ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']
    regression_feature_set_5 = ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']
    
    
#    train_full = train_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']]
    train_full = train_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f292', 'f73', 'f419', 'f277', 'f298']]
    test_full  = test_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f292', 'f73', 'f419', 'f277', 'f298']]
   
   
   
#    train_full = train_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f292', 'f73', 'f419', 'f277', 'f298', 'f377', 'f112', 'f746', 'f135', 'f428', 'f667', 'f336', 'f272', 'f239', 'f27', 'f499', 'f775', 'f233', 'f334', 'f337', 'f418', 'f241', 'f335', 'f338', 'f648', 'f81', 'f471', 'f518', 'f679', 'f635', 'f16', 'f69', 'f617', 'f148', 'f164', 'f588', 'f25', 'f765', 'f556', 'f83']]

#    test_full = test_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']]
        
    ## Best Spearman ##
#    train_full = train_full[['f527', 'f274', 'f776', 'f1', 'f619', 'f5', 'f650', 'f216']]
    
    imp = Imputer()
    imp.fit(train_full)    
    train_full = imp.transform(train_full)
    test_full = imp.transform(test_full)
    
    scaler = pre.StandardScaler()
    scaler.fit(train_full)
    train_full=scaler.transform(train_full)
    test_full=scaler.transform(test_full) 
#    
    train = train[binary_classification_feature_set_2]
    test = test[binary_classification_feature_set_2]
    
#    print train.shape

#    train = train[['f527', 'f528', 'f274', 'f271']]
#    test = test[['f527', 'f528', 'f274', 'f271']]
    
    imp = Imputer()
    imp.fit(train)
    
    train = imp.transform(train)
    test = imp.transform(test)
    
    scaler = pre.StandardScaler()
    scaler.fit(train)
    train=scaler.transform(train)
    test=scaler.transform(test)    
    
#    print train.shape
#    train = train[[True] * train.shape[0]]
#    print train[0]
#    test = test[[True] * test.shape[0]]
#    
    y_train_initial = train_loss
    y_train = train_loss.apply(lambda x: 1 if x>0 else 0).values
#    
#    y_train = y_train[[True] * train.shape[0]]
#    
#    print y_train[10:30]
#    print train_loss.values[10:30]
#
#    X_train,X_test,y_train,y_test = train_test_split( train, train_loss.apply(lambda x: 1 if x>0 else 0), test_size=0.3, random_state=42)
#    X_trainss,X_testss,y_trainss,y_testss = train_test_split( test, np.zeros(shape=(test.shape[0])), test_size=0., random_state=42)
#    
    clf = LogisticRegression(C=1e20,penalty='l2')
    
    clf.fit(train,y_train)      
    probas_ = clf.predict_proba(test)
        
    prediction_proba = probas_[:,1]
#          

#    prediction_proba = clf.predict_proba(test)
    
    predictionIndexes0 = np.where(prediction_proba <= average_best_t)[0]
    predictionIndexes1 = np.where(prediction_proba > average_best_t)[0]     
    
    train_non_zero_indexes = np.where(y_train > 0)[0]
    train_zero_indexes = np.where(y_train == 0)[0]
    
    print "Prediction counts: .. " + str(len(predictionIndexes1)) + " " + str(len(predictionIndexes0)) + " " + str(len(predictionIndexes0) + len(predictionIndexes1)) 
    
    print "Train counts: " + str(len(train_non_zero_indexes)) + " " + str(len(train_zero_indexes))   
    
    X_train_full_split_non_zero = train_full[train_non_zero_indexes]
    y_train_split_non_zero = y_train_initial[train_non_zero_indexes]
    
    X_test_full_split_non_zero = test_full[predictionIndexes1]
    
    regr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
#    regr = SVR(kernel='rbf', degree=2, gamma=0.0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, probability=False, cache_size=200, verbose=False, max_iter=-1, random_state=None)
#
#    regr.fit(X_train_full_split_non_zero, y_train_split_non_zero)   
#    predicted_regr_values = regr.predict(X_test_full_split_non_zero)


#                                  
#        regr.fit(X_train_full_split_non_zero, y_train_split_non_zero)
        
#        predicted_regr_values = regr.predict(X_test_full_split_non_zero)

    beta = quantilereg(y_train_split_non_zero,X_train_full_split_non_zero,0.25)
    print beta
    predicted_regr_values = []
    for x_t in X_test_full_split_non_zero:
#            print len(beta)
        pred = np.dot(x_t, beta[1:]) + beta[0]
#            print pred
        
        predicted_regr_values += [pred]
    predicted_regr_values_norm = np.zeros(shape=(len(predicted_regr_values)))
    predicted_regr_values_norm = np.zeros(shape=(len(predicted_regr_values)))
    for i, x in enumerate(predicted_regr_values):
        if x < 0:
            predicted_regr_values_norm[i] = 0.
        elif x > 100:
            predicted_regr_values_norm[i] = 100.
        else:
            predicted_regr_values_norm[i] = x
    predicted_regr_values = predicted_regr_values_norm
#    predicted_regr_values = predicted_regr_values / 3.
    
    
    
    print predicted_regr_values
    
    final_prediction = np.zeros(shape=(test.shape[0]))
    
    final_prediction[predictionIndexes1] = predicted_regr_values
    final_prediction[predictionIndexes0] = 0.
    
    np.savetxt('predictions/predictions.csv',final_prediction ,delimiter = ',')
    
    prediction_values_file = "predictions/predictions.csv"
    sample_file = "data/sampleSubmission.csv"
    competition_format_file = "predictions/predictions_competition_ensemble_run_1.csv"
    write_prediction_file(prediction_values_file, sample_file, competition_format_file)
    
    

def make_prediction_2():

    use_sample = False
    
    if use_sample:
        train = pandas.read_csv('data/train_v2_sample_10k.csv')
        test = pandas.read_csv('data/test_v2_sample_10k.csv')
        average_best_t = 0.148846575958
    else:
        train = pandas.read_csv('data/train_v2.csv')
        test = pandas.read_csv('data/test_v2.csv')
         ### To use on full train set
        average_best_t = 0.164463473639
  
    
    train_loss = train.loss
    train_full = train.copy()
    test_full = test.copy()   
    
    binary_classification_feature_set_1 = ['f527', 'f528', 'f274', 'f271']
    binary_classification_feature_set_2 = ['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431', 'f757']
    
    regression_feature_set_1 = ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']
    regression_feature_set_2 = ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']
    regression_feature_set_3 = ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']
    regression_feature_set_4 = ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']
    regression_feature_set_5 = ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']
    
    
#    train_full = train_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']]
    train_full = train_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f292', 'f73', 'f419', 'f277', 'f298']]
    test_full  = test_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f292', 'f73', 'f419', 'f277', 'f298']]
   
      
#    train_full = train_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f292', 'f73', 'f419', 'f277', 'f298', 'f377', 'f112', 'f746', 'f135', 'f428', 'f667', 'f336', 'f272', 'f239', 'f27', 'f499', 'f775', 'f233', 'f334', 'f337', 'f418', 'f241', 'f335', 'f338', 'f648', 'f81', 'f471', 'f518', 'f679', 'f635', 'f16', 'f69', 'f617', 'f148', 'f164', 'f588', 'f25', 'f765', 'f556', 'f83']]

#    test_full = test_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']]
        
    ## Best Spearman ##
#    train_full = train_full[['f527', 'f274', 'f776', 'f1', 'f619', 'f5', 'f650', 'f216']]
    
    imp = Imputer()
    imp.fit(train_full)    
    train_full = imp.transform(train_full)
    test_full = imp.transform(test_full)
    
    scaler = pre.StandardScaler()
    scaler.fit(train_full)
    train_full=scaler.transform(train_full)
    test_full=scaler.transform(test_full) 
#    
    train = train[binary_classification_feature_set_2]
    test = test[binary_classification_feature_set_2]

    
    imp = Imputer()
    imp.fit(train)
    
    train = imp.transform(train)
    test = imp.transform(test)
    
    scaler = pre.StandardScaler()
    scaler.fit(train)
    train=scaler.transform(train)
    test=scaler.transform(test)    

    y_train_initial = train_loss
    y_train = train_loss.apply(lambda x: 1 if x>0 else 0).values

#    
    clf = LogisticRegression(C=1e20,penalty='l2')
    
    clf.fit(train,y_train)      
    probas_ = clf.predict_proba(test)
        
    prediction_proba = probas_[:,1]

    
    predictionIndexes0 = np.where(prediction_proba <= average_best_t)[0]
    predictionIndexes1 = np.where(prediction_proba > average_best_t)[0]     
    
    train_non_zero_indexes = np.where(y_train > 0)[0]
    train_zero_indexes = np.where(y_train == 0)[0]
    
    print "Prediction counts: .. " + str(len(predictionIndexes1)) + " " + str(len(predictionIndexes0)) + " " + str(len(predictionIndexes0) + len(predictionIndexes1)) 
    
    print "Train counts: " + str(len(train_non_zero_indexes)) + " " + str(len(train_zero_indexes))   
    
    X_train_full_split_non_zero = train_full[train_non_zero_indexes]
    y_train_split_non_zero = y_train_initial[train_non_zero_indexes]
    
    X_test_full_split_non_zero = test_full[predictionIndexes1]
    
    kf = StratifiedKFold(y_train_split_non_zero, n_folds=5, indices=False)
    X_predicted_regr_values_quantile = np.zeros(shape=(len(y_train_split_non_zero), 4))
    y_actual_regr_values = np.zeros(shape=(len(y_train_split_non_zero)))
    for train_i, test_i in kf:
#        print len(train_i)
        X_train_split, X_test_split, y_train_split, y_test_split = X_train_full_split_non_zero[train_i], X_train_full_split_non_zero[test_i], y_train_split_non_zero[train_i], y_train_split_non_zero[test_i]
        
        predicted_regr_values_02 = predict_quantile_regr(y_train_split, X_train_split, X_test_split, 0.2)
        print "DONE QREG 1"
        predicted_regr_values_04 = predict_quantile_regr(y_train_split, X_train_split, X_test_split, 0.4)
        print "DONE QREG 2"
        predicted_regr_values_06 = predict_quantile_regr(y_train_split, X_train_split, X_test_split, 0.6)
        print "DONE QREG 3"
        predicted_regr_values_08 = predict_quantile_regr(y_train_split, X_train_split, X_test_split, 0.8)
        print "DONE QREG 4"    

        predicted_regr_values_X = [[p1, p2, p3, p4] for p1, p2, p3, p4 in zip(predicted_regr_values_02, predicted_regr_values_04, 
                               predicted_regr_values_06, predicted_regr_values_08)]
        predicted_regr_values_X = np.array(predicted_regr_values_X)

        print predicted_regr_values_X.shape        
        
        predicted_indeces = np.where(test_i == True)[0] 
        X_predicted_regr_values_quantile[predicted_indeces] = predicted_regr_values_X
        y_actual_regr_values[predicted_indeces] = y_test_split    

    print X_predicted_regr_values_quantile.shape
    print y_actual_regr_values.shape
    regr_predictor_ensemble = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=500, 
                              subsample=1.0, min_samples_split=2, min_samples_leaf=1, 
                              max_depth=3, init=None, random_state=None, max_features=None, 
                              alpha=0.9, verbose=0)
    regr_predictor_ensemble.fit(X_predicted_regr_values_quantile, y_actual_regr_values)
    
    print "Train individual regressors on full train set"
    
    predicted_regr_values_02 = predict_quantile_regr(y_train_split_non_zero, X_train_full_split_non_zero, X_test_full_split_non_zero, 0.2)
    print "DONE QREG 1"
    predicted_regr_values_04 = predict_quantile_regr(y_train_split_non_zero, X_train_full_split_non_zero, X_test_full_split_non_zero, 0.4)
    print "DONE QREG 2"
    predicted_regr_values_06 = predict_quantile_regr(y_train_split_non_zero, X_train_full_split_non_zero, X_test_full_split_non_zero, 0.6)
    print "DONE QREG 3"
    predicted_regr_values_08 = predict_quantile_regr(y_train_split_non_zero, X_train_full_split_non_zero, X_test_full_split_non_zero, 0.8)
    predicted_regr_values_test = [[p1, p2, p3, p4] for p1, p2, p3, p4 in zip(predicted_regr_values_02, predicted_regr_values_04, 
                               predicted_regr_values_06, predicted_regr_values_08)]
    predicted_regr_values_test = np.array(predicted_regr_values_test)
    
    print "Predict ensemble for test" 
    
    predicted_regr_values = regr_predictor_ensemble.predict(predicted_regr_values_test)                           

    
    predicted_regr_values_norm = np.zeros(shape=(len(predicted_regr_values)))
    for i, x in enumerate(predicted_regr_values):
        if x < 0:
            predicted_regr_values_norm[i] = 0.
        elif x > 100:
            predicted_regr_values_norm[i] = 100.
        else:
            predicted_regr_values_norm[i] = x
    predicted_regr_values = predicted_regr_values_norm
    
    print predicted_regr_values
    
    final_prediction = np.zeros(shape=(test.shape[0]))
    
    final_prediction[predictionIndexes1] = predicted_regr_values
    final_prediction[predictionIndexes0] = 0.
    
    np.savetxt('predictions/predictions.csv',final_prediction ,delimiter = ',')
    
    prediction_values_file = "predictions/predictions.csv"
    sample_file = "data/sampleSubmission.csv"
    competition_format_file = "predictions/predictions_competition_ensemble_run_2.csv"
    write_prediction_file(prediction_values_file, sample_file, competition_format_file)


def make_prediction_3():

    use_sample = False
    
    if use_sample:
        train = pandas.read_csv('data/train_v2_sample_10k.csv')
        test = pandas.read_csv('data/test_v2_sample_10k.csv')
        average_best_t = 0.148846575958
    else:
        train = pandas.read_csv('data/train_v2.csv')
        test = pandas.read_csv('data/test_v2.csv')
         ### To use on full train set
        average_best_t = 0.164463473639
  
    
    train_loss = train.loss
    train_full = train.copy()
    test_full = test.copy()   
    
    binary_classification_feature_set_1 = ['f527', 'f528', 'f274', 'f271']
    binary_classification_feature_set_2 = ['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431', 'f757']
    
    regression_feature_set_1 = ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']
    regression_feature_set_2 = ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']
    regression_feature_set_3 = ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']
    regression_feature_set_4 = ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']
    regression_feature_set_5 = ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']
    
    
#    train_full = train_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']]
    train_full = train_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f292', 'f73', 'f419', 'f277', 'f298']]
    test_full  = test_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f292', 'f73', 'f419', 'f277', 'f298']]
   
      
#    train_full = train_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f292', 'f73', 'f419', 'f277', 'f298', 'f377', 'f112', 'f746', 'f135', 'f428', 'f667', 'f336', 'f272', 'f239', 'f27', 'f499', 'f775', 'f233', 'f334', 'f337', 'f418', 'f241', 'f335', 'f338', 'f648', 'f81', 'f471', 'f518', 'f679', 'f635', 'f16', 'f69', 'f617', 'f148', 'f164', 'f588', 'f25', 'f765', 'f556', 'f83']]

#    test_full = test_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']]
        
    ## Best Spearman ##
#    train_full = train_full[['f527', 'f274', 'f776', 'f1', 'f619', 'f5', 'f650', 'f216']]
    
    imp = Imputer()
    imp.fit(train_full)    
    train_full = imp.transform(train_full)
    test_full = imp.transform(test_full)
    
    scaler = pre.StandardScaler()
    scaler.fit(train_full)
    train_full=scaler.transform(train_full)
    test_full=scaler.transform(test_full) 
#    
    train = train[binary_classification_feature_set_2]
    test = test[binary_classification_feature_set_2]

    
    imp = Imputer()
    imp.fit(train)
    
    train = imp.transform(train)
    test = imp.transform(test)
    
    scaler = pre.StandardScaler()
    scaler.fit(train)
    train=scaler.transform(train)
    test=scaler.transform(test)    

    y_train_initial = train_loss
    y_train = train_loss.apply(lambda x: 1 if x>0 else 0).values

#    
    clf = LogisticRegression(C=1e20,penalty='l2')
    
    clf.fit(train,y_train)      
    probas_ = clf.predict_proba(test)
        
    prediction_proba = probas_[:,1]

    
    predictionIndexes0 = np.where(prediction_proba <= average_best_t)[0]
    predictionIndexes1 = np.where(prediction_proba > average_best_t)[0]     
    
    train_non_zero_indexes = np.where(y_train > 0)[0]
    train_zero_indexes = np.where(y_train == 0)[0]
    
    print "Prediction counts: .. " + str(len(predictionIndexes1)) + " " + str(len(predictionIndexes0)) + " " + str(len(predictionIndexes0) + len(predictionIndexes1)) 
    
    print "Train counts: " + str(len(train_non_zero_indexes)) + " " + str(len(train_zero_indexes))   
    
    X_train_full_split_non_zero = train_full[train_non_zero_indexes]
    y_train_split_non_zero = y_train_initial[train_non_zero_indexes]
    
    X_test_full_split_non_zero = test_full[predictionIndexes1]
    
    kf = StratifiedKFold(y_train_split_non_zero, n_folds=5, indices=False)
    X_predicted_regr_values_quantile = np.zeros(shape=(len(y_train_split_non_zero), 3))
    y_actual_regr_values = np.zeros(shape=(len(y_train_split_non_zero)))
    for train_i, test_i in kf:
#        print len(train_i)
        X_train_split, X_test_split, y_train_split, y_test_split = X_train_full_split_non_zero[train_i], X_train_full_split_non_zero[test_i], y_train_split_non_zero[train_i], y_train_split_non_zero[test_i]
        
        predicted_regr_values_25 = predict_quantile_regr(y_train_split, X_train_split, X_test_split, 0.25)
        print "DONE QREG 1"
        predicted_regr_values_50 = predict_quantile_regr(y_train_split, X_train_split, X_test_split, 0.5)
        print "DONE QREG 2"
        predicted_regr_values_75 = predict_quantile_regr(y_train_split, X_train_split, X_test_split, 0.75)
        print "DONE QREG 3"         

        predicted_regr_values_X = [[p1, p2, p3] for p1, p2, p3 in zip(predicted_regr_values_25, predicted_regr_values_50, 
                               predicted_regr_values_75)]
        predicted_regr_values_X = np.array(predicted_regr_values_X)

        print predicted_regr_values_X.shape        
        
        predicted_indeces = np.where(test_i == True)[0] 
        X_predicted_regr_values_quantile[predicted_indeces] = predicted_regr_values_X
        y_actual_regr_values[predicted_indeces] = y_test_split    

    print X_predicted_regr_values_quantile.shape
    print y_actual_regr_values.shape
    regr_predictor_ensemble = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=500, 
                              subsample=1.0, min_samples_split=2, min_samples_leaf=1, 
                              max_depth=3, init=None, random_state=None, max_features=None, 
                              alpha=0.9, verbose=0)
                              
#    regr_predictor_ensemble = SVR(kernel='rbf', degree=2, gamma=0.0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, probability=False, cache_size=200, verbose=False, max_iter=-1, random_state=None)
    
    regr_predictor_ensemble.fit(X_predicted_regr_values_quantile, y_actual_regr_values)
    
    print "Train individual regressors on full train set"
    
    predicted_regr_values_25 = predict_quantile_regr(y_train_split_non_zero, X_train_full_split_non_zero, X_test_full_split_non_zero, 0.25)
    print "DONE QREG 1"
    predicted_regr_values_50 = predict_quantile_regr(y_train_split_non_zero, X_train_full_split_non_zero, X_test_full_split_non_zero, 0.5)
    print "DONE QREG 2"
    predicted_regr_values_75 = predict_quantile_regr(y_train_split_non_zero, X_train_full_split_non_zero, X_test_full_split_non_zero, 0.75)
    print "DONE QREG 3"
    
    predicted_regr_values_test =  [[p1, p2, p3] for p1, p2, p3 in zip(predicted_regr_values_25, predicted_regr_values_50, 
                               predicted_regr_values_75)]
#    predicted_regr_values_test = np.array(predicted_regr_values_test)
    
    print "Predict ensemble for test" 
    
    predicted_regr_values = regr_predictor_ensemble.predict(predicted_regr_values_test)                          

    
    predicted_regr_values_norm = np.zeros(shape=(len(predicted_regr_values)))
    for i, x in enumerate(predicted_regr_values):
        if x < 0:
            predicted_regr_values_norm[i] = 0.
        elif x > 100:
            predicted_regr_values_norm[i] = 100.
        else:
            predicted_regr_values_norm[i] = x
    predicted_regr_values = predicted_regr_values_norm
    
    print predicted_regr_values
    
    final_prediction = np.zeros(shape=(test.shape[0]))
    
    final_prediction[predictionIndexes1] = predicted_regr_values
    final_prediction[predictionIndexes0] = 0.
    
    np.savetxt('predictions/predictions.csv',final_prediction ,delimiter = ',')
    
    prediction_values_file = "predictions/predictions.csv"
    sample_file = "data/sampleSubmission.csv"
    competition_format_file = "predictions/predictions_competition_ensemble_run_3.csv"
    write_prediction_file(prediction_values_file, sample_file, competition_format_file)


def make_prediction_4():

    use_sample = False
    
    if use_sample:
        train = pandas.read_csv('data/train_v2_sample_10k.csv')
        test = pandas.read_csv('data/test_v2_sample_10k.csv')
       
    else:
        train = pandas.read_csv('data/train_v2.csv')
        test = pandas.read_csv('data/test_v2.csv')
         ### To use on full train set      
  
    
    train_loss = train.loss
    
    cols = set(train.columns)
    cols.remove('loss')
    cols = list(cols)
    train = train[cols]   
           
   
    imp = Imputer()
    imp.fit(train)
    
    train = imp.transform(train)
    test = imp.transform(test)
    
    scaler = pre.StandardScaler()
    scaler.fit(train)
    train=scaler.transform(train)
    test=scaler.transform(test)    

    train_loss_array = train_loss.apply(lambda x: 1 if x>0 else 0).values
    
    clf = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.1, n_jobs=6, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, rho=None, seed=None)
           
    clf.fit(train,train_loss_array)      
    train = clf.transform(train, threshold = "1.25*mean")
    test = clf.transform(test, threshold = "1.25*mean")
    print train.shape    
    print test.shape
    
    threshold  = 0.999999999164 
    
    clf = SGDClassifier(loss='log', penalty='l2', alpha=1e-4, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, epsilon=0.1, n_jobs=6, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, rho=None, seed=None)

    clf.fit(train,train_loss_array)      
    probas_ = clf.predict_proba(test)
    prediction_proba = probas_[:,1]
    
    predictionIndexes0 = np.where(prediction_proba <= threshold)[0]
    predictionIndexes1 = np.where(prediction_proba > threshold)[0]
    
    print predictionIndexes1.shape
    
    prediction = np.asarray([0.] * test.shape[0])
    prediction[predictionIndexes1] = 10.
    prediction[predictionIndexes0] = 0.   
        
    np.savetxt('predictions/predictions.csv',prediction ,delimiter = ',')
    
    prediction_values_file = "predictions/predictions.csv"
    sample_file = "data/sampleSubmission.csv"
    competition_format_file = "predictions/SGD_binary_10_rest.csv"
    write_prediction_file(prediction_values_file, sample_file, competition_format_file)

    
if __name__ == "__main__":
    make_prediction_4()

    
    
