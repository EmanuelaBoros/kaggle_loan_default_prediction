import pandas

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn import cross_validation

from sklearn.preprocessing import Imputer

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn import preprocessing as pre
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor
from sklearn import svm, datasets
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import f_regression
from sklearn.svm import SVR
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
 
    #_______estimating variances based on Green 2008(quantile regression)______
 
    e=y-dot(x,beta)
    iqre=np.percentile(e,0.75)-np.percentile(e,0.25)
    if p==0.5:
        h=0.9*np.std(e)/(ry**0.2)
    else:
        h=0.9*np.min(np.std(e),iqre/1.34)/(ry**0.2)
    u=(e/h)
    fhat0=(1/(ry*h))*(sum(exp(-u)/((1+exp(-u))**2)))
    D=np.zeros((ry,ry))
    DIAGON=np.diag(D)
    DIAGON[e>0]=(p/fhat0)**2
    DIAGON[e<=0]=((1-p)/fhat0)**2
    D=np.diag(DIAGON)
    VCQ=np.dot(inv(dot(x.T,x)),dot(x.T,D,x),inv(np.dot(x.T,x)))
 
    #____________________Standarad errores and t-stats_________________________
 
    tstats=beta/np.sqrt(np.diag(VCQ))
    stderrors=np.sqrt(np.diag(VCQ))
    PValues=2*(1-stats.t.cdf(np.abs(tstats),ry-cx))
 
    #______________________________ Quasi R square_____________________________
 
    ef=y-dot(x,beta)
    ef[ef<0]=(1-p)*ef[ef<0]
    ef[ef>0]=p*ef[ef>0]
    ef=np.abs(ef)
 
    ered=y-np.percentile(y,p)
    ered[ered<0]=(1-p)*ered[ered<0]
    ered[ered>0]=p*ered[ered>0]
    ered=np.abs(ered)
 
    PseudoR2=1-np.sum(ef)/np.sum(ered)
 
    #__________________Bootstrap standard deviation (Green 2008)_______________
 
    betaboot=np.zeros((cx,cx))
    for ii in range(100):
        bootm, estar=bootstrp(1,np.mean,e)
        #
        ystar=dot(x,beta)+e[estar]
        #
        itratstar=0
        xstarstar=x
        diffstar=1
        betastar=np.ones(cx)
        while itratstar<1000 and diffstar>1e-6:
            itratstar=itratstar+1
            betastar0=betastar
            betastar=dot(inv(dot(xstarstar.T,x)),xstarstar.T,ystar)
            #
            residstar=ystar-dot(x,betastar)
            residstar[np.abs(residstar)<.000001]=.000001
            residstar[residstar<0]=p*residstar[residstar<0]
            residstar[residstar>0]=(1-p)*residstar[residstar>0]
            residstar=np.abs(residstar)
            zstar=np.zeros((rx,cx))
            for i in range(cx):
                zstar[:,i] = x[:,i]/residstar
            xstarstar=zstar
            beta1star=betastar
            diffstar=np.max(np.abs(beta1star-betastar0))
        #
        betaboot=[betaboot + dot((betastar-beta),(betastar-beta).T)]
    VCboot=(1/100)*betaboot
    #
    tstatsboot=beta/diag(VCboot)**0.5
    stderrorsboot=diag(VCboot)**0.5
    PValuesboot=2*(1-stats.t.cdf(np.abs(tstatsboot),ry-cx))
 
    #_______________________________Display Results____________________________
 
    print
    print(' Results of Quantile Regression')
    print('_'*70)
    print("%10s %10s %10s %10s %10s %10s %10s" % ['Coef.', 'SE.Ker', 't.Ker', 'P.Ker', 'SE.Boot', 't.Boot', 'P.Boot'])
    print('_'*70)
    print("%10f %10f %10f %10f %10f %10f %10f" % [ beta,stderrors,tstats,PValues,stderrorsboot,tstatsboot,PValuesboot])
    print('_'*70)
    print('Pseudo R2: %10f' % PseudoR2 )
    print('_'*70)

def test_regression_models():

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
    
    pickle.dump( kf, open( "data/LogisticRegression_StratifiedKFold.pkl", "wb" ) )
   

    
    print "Average best threshold " + str(average_best_t)
    
    print "Calculate MAE with average best threshold "
    average_mae = 0.
    average_pearson = 0.
    split = 0
    for train_i, test_i in kf:
        split += 1
#        print len(train_i)
        X_train_split, X_test_split, y_train_split, y_test_split = train[train_i], train[test_i], train_loss_array[train_i], train_loss_array[test_i]
        y_test_split_initial = train_loss[test_i].values
        y_train_split_initial = train_loss[train_i].values        

        
        clf = LogisticRegression(C=1e20,penalty='l2')
    
        clf.fit(X_train_split,y_train_split)      
        probas_ = clf.predict_proba(X_test_split)
        
#        print "... " + str(len(y_test_split_initial)) + " " + str(len(probas_))
        
        prediction_proba = probas_[:,1]
        
        pickle.dump( probas_, open( "data/LogisticRegression_probas_split_" + str(split) + ".pkl", "wb" ) )
        
        predictionIndexes0 = np.where(prediction_proba <= average_best_t)[0]
        predictionIndexes1 = np.where(prediction_proba > average_best_t)[0]       
           
#        print "Prediction counts: .. " + str(len(predictionIndexes1)) + " " + str(len(predictionIndexes0)) + " " + str(len(predictionIndexes0) + len(predictionIndexes1)) 
        
        train_non_zero_indexes = np.where(y_train_split_initial > 0)[0]
#        train_zero_indexes = np.where(y_train_split_initial == 0)[0]
#        print "Train counts: " + str(len(train_non_zero_indexes)) + " " + str(len(train_zero_indexes))
        
        X_train_split_non_zero = X_train_split[train_non_zero_indexes]
        y_train_split_non_zero = y_train_split_initial[train_non_zero_indexes]
        
        X_test_split_non_zero = X_test_split[predictionIndexes1]
        y_test_split_initial_non_zero = y_test_split_initial[predictionIndexes1]
        
#        y_test_split_initial_non_zero_actual =  np.where(y_test_split_initial > 0)[0]
#        y_test_split_initial_zero_actual =  np.where(y_test_split_initial == 0)[0]
        
#        print "Real test counts: " + str(len(y_test_split_initial_non_zero_actual)) + " " + str(len(y_test_split_initial_zero_actual))
        
        ##############" GradientBoostingRegressor ##########################
        regr = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=500, 
                                  subsample=1.0, min_samples_split=2, min_samples_leaf=1, 
                                  max_depth=3, init=None, random_state=None, max_features=None, 
                                  alpha=0.9, verbose=0)
                                  
        ### Best MAE : 0.714 with learning_rate=0.1, n_estimators=500,  predictions / 3 ###
                                  
        ##############" LinearRegression ##########################
        
        regr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
        
        ##############" Ridge ##########################
        
#        regr = Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.01, solver='auto')
        
        ##############" LogisticRegression ##########################
        
#        regr = LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
#                             C=1e20, fit_intercept=True, intercept_scaling=1.0, 
#                             class_weight=None, random_state=None)
                                  
        print str(split) + "... " 
#        print X_train_split_non_zero[0]
        regr.fit(X_train_split_non_zero, y_train_split_non_zero)
        
        predicted_regr_values = regr.predict(X_test_split_non_zero)
        print predicted_regr_values[:10]
        print y_test_split_initial_non_zero[:10]       
        
        pearson = pearsonr(predicted_regr_values, y_test_split_initial_non_zero)[0]
        
        print X_test_split_non_zero[0]
        
        average_pearson += pearson
        print "Pearson : " + str(pearson)

        predicted_regr_values = predicted_regr_values / 3.
            
        prediction = np.asarray([0.] * y_test_split_initial.shape[0])
        prediction[predictionIndexes1] = predicted_regr_values
        prediction[predictionIndexes0] = 0.
        mae = mean_absolute_error(y_test_split_initial, prediction)
        print "MAE :" + str(mae)      
        average_mae += mae                          
        
                  
    average_mae = average_mae / 5.
    average_pearson = average_pearson / 5.
    print "Mean MAE: " + str(average_mae)   
    print "Mean Pearson: " + str(average_pearson) 
          
 
def test_regression_models_all_features():

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

    train_full = train.copy()
#    test_full = test.copy()   
    
#    cols = set(train_full.columns)
#    cols.remove('loss')
#    cols = list(cols)
#    train_full = train_full[cols]
    
    train_full = train_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']]
#    train_full = train_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f292', 'f73', 'f419', 'f277', 'f298']]
   
#    train_full = train_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f292', 'f73', 'f419', 'f277', 'f298', 'f377', 'f112', 'f746', 'f135', 'f428', 'f667', 'f336', 'f272', 'f239', 'f27', 'f499', 'f775', 'f233', 'f334', 'f337', 'f418', 'f241', 'f335', 'f338', 'f648', 'f81', 'f471', 'f518', 'f679', 'f635', 'f16', 'f69', 'f617', 'f148', 'f164', 'f588', 'f25', 'f765', 'f556', 'f83']]

#    test_full = test_full[['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f73']]
    
    
    
    imp = Imputer()
    imp.fit(train_full)
    
    train_full = imp.transform(train_full)
#    test_full = imp.transform(test_full)
    
    train_full=pre.StandardScaler().fit_transform(train_full)
#    test_full=pre.StandardScaler().fit_transform(test_full) 

    print "Selecting regression features: "
#    train_non_zero_indexes = np.where(train_loss.values > 0)[0]      
#    
#    featureSelector = SelectKBest(score_func=f_regression,k=10)
#    featureSelector.fit(train_full[train_non_zero_indexes],train_loss.values[train_non_zero_indexes])
#    print "Selected features: " 
#    print [1+zero_based_index for zero_based_index in list(featureSelector.get_support(indices=True))]

   
    
    
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
   

    
    print "Average best threshold " + str(average_best_t)
    
    print "Calculate MAE with average best threshold "
    average_mae = 0.
    average_pearson = 0.
    for train_i, test_i in kf:
#        print len(train_i)
         
        X_train_split, X_test_split, y_train_split, y_test_split = train[train_i], train[test_i], train_loss_array[train_i], train_loss_array[test_i]
        y_test_split_initial = train_loss[test_i].values
        y_train_split_initial = train_loss[train_i].values
        
        X_train_full_split, X_test_full_split = train_full[train_i], train_full[test_i]        

        
        clf = LogisticRegression(C=5,penalty='l2')
    
        clf.fit(X_train_split,y_train_split)      
        probas_ = clf.predict_proba(X_test_split)
        
#        print "... " + str(len(y_test_split_initial)) + " " + str(len(probas_))
        
        prediction_proba = probas_[:,1]
        
        predictionIndexes0 = np.where(prediction_proba <= average_best_t)[0]
        predictionIndexes1 = np.where(prediction_proba > average_best_t)[0]       
           
        print "Prediction counts: .. " + str(len(predictionIndexes1)) + " " + str(len(predictionIndexes0)) + " " + str(len(predictionIndexes0) + len(predictionIndexes1)) 
        
        train_non_zero_indexes = np.where(y_train_split_initial > 0)[0]
        train_zero_indexes = np.where(y_train_split_initial == 0)[0]
        print "Train counts: " + str(len(train_non_zero_indexes)) + " " + str(len(train_zero_indexes))
        
#        X_train_split_non_zero = X_train_split[train_non_zero_indexes]
        y_train_split_non_zero = y_train_split_initial[train_non_zero_indexes]
        
        X_train_full_split_non_zero = X_train_full_split[train_non_zero_indexes]
        
#        X_test_split_non_zero = X_test_split[predictionIndexes1]
        y_test_split_initial_non_zero = y_test_split_initial[predictionIndexes1]
        
        X_test_full_split_non_zero = X_test_full_split[predictionIndexes1]
        
#        y_test_split_initial_non_zero_actual =  np.where(y_test_split_initial > 0)[0]
#        y_test_split_initial_zero_actual =  np.where(y_test_split_initial == 0)[0]
        
#        print "Real test counts: " + str(len(y_test_split_initial_non_zero_actual)) + " " + str(len(y_test_split_initial_zero_actual))

#        X_train_full_split_non_zero = featureSelector.transform(X_train_full_split_non_zero)
#        X_test_full_split_non_zero = featureSelector.transform(X_test_full_split_non_zero)
        
        ##############" GradientBoostingRegressor ##########################
        regr = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=500, 
                                  subsample=1.0, min_samples_split=2, min_samples_leaf=1, 
                                  max_depth=3, init=None, random_state=None, max_features=None, 
                                  alpha=0.9, verbose=0)
                                  
        ### Best MAE : 0.714 with learning_rate=0.1, n_estimators=500,  predictions / 3 ###
                                  
        ##############" LinearRegression ##########################
        
        regr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
       
        ##############" Ridge ##########################
        
#        regr = Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.01, solver='auto')

        regr = linear_model.RidgeCV(alphas=[1E-10, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])

        ##############" GaussianRegressor ##########################
                
        regr = GaussianRegressor() 
        ######  Mean Pearson: 0.430804442235    MAE: 0.654458711488   #####""
         ###### ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f292', 'f73', 'f419', 'f277', 'f298']   #####""
        
        ##############" GaussianRegressor ##########################
                
        regr = PoissonRegressor() 
        
         ######  Mean Pearson: 0.432911478313    MAE: 0.659572430061   #####""
         ###### ['f527', 'f528', 'f404', 'f67', 'f230', 'f734', 'f2', 'f421', 'f120', 'f26', 'f670', 'f598', 'f292', 'f73', 'f419', 'f277', 'f298']   #####""
        
        regr = SVR(kernel='rbf', degree=3, gamma=0.0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, probability=False, cache_size=200, verbose=False, max_iter=-1, random_state=None)
        ######  Mean Pearson: 0.4633868536593 ########
        regr = SVR(kernel='rbf', degree=2, gamma=0.0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, probability=False, cache_size=200, verbose=False, max_iter=-1, random_state=None)
        ######  Mean Pearson: 0.463386853659 ########
#        regr = SVR(kernel='rbf', degree=4, gamma=0.0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, probability=False, cache_size=200, verbose=False, max_iter=-1, random_state=None)
        ######  Mean Pearson: 0.463386853659 ########
#        regr = SVR(kernel='rbf', degree=2, gamma=0.0, coef0=0.0, tol=0.001, C=10.0, epsilon=0.1, shrinking=True, probability=False, cache_size=200, verbose=False, max_iter=-1, random_state=None)
        ######  Mean Pearson: 0.459173633354 ########
#        regr = SVR(kernel='poly', degree=3, gamma=0.1, coef0=0.0, tol=0.001, C=1e3, epsilon=0.1, shrinking=True, probability=False, cache_size=200, verbose=False, max_iter=-1, random_state=None)
        #####  Mean Pearson: ? ########
#        regr = SVR(kernel='rbf', degree=3, gamma=0.1, coef0=0.0, tol=0.001, C=1e3, epsilon=0.1, shrinking=True, probability=False, cache_size=200, verbose=False, max_iter=-1, random_state=None)
        #####  Mean Pearson: 0.bad ########
#        regr = SVR(kernel='rbf', degree=3, gamma=0.0, coef0=0.0, tol=0.001, C=1e-1, epsilon=0.1, shrinking=True, probability=False, cache_size=200, verbose=False, max_iter=-1, random_state=None)
        #####  Mean Pearson: 0.363158006809 ########
        
#        beta = quantilereg(y_train_split_non_zero,X_train_full_split_non_zero,0.9)
#        print beta
#        predicted_regr_values = []
#        for x_t in X_test_full_split_non_zero:
##            print len(beta)
#            pred = np.dot(x_t, beta[1:]) + beta[0]
##            print pred
#            
#            predicted_regr_values += [pred]
#                                  
        regr.fit(X_train_full_split_non_zero, y_train_split_non_zero)
        
        predicted_regr_values = regr.predict(X_test_full_split_non_zero)
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
                    
        print predicted_regr_values[:10]
        print y_test_split_initial_non_zero[:10]      
        
        pearson = pearsonr(predicted_regr_values, y_test_split_initial_non_zero)[0]
#        pearson =  spearmanr(predicted_regr_values, y_test_split_initial_non_zero)[0]
        average_pearson += pearson
        print "Pearson : " + str(pearson)

        predicted_regr_values = predicted_regr_values / 1.
#        predicted_regr_values = predicted_regr_values / 1.
            
        prediction = np.asarray([0.] * y_test_split_initial.shape[0])
        prediction[predictionIndexes1] = predicted_regr_values
        prediction[predictionIndexes0] = 0.
        mae = mean_absolute_error(y_test_split_initial, prediction)
        print "MAE :" + str(mae)      
        average_mae += mae          

#        pl.scatter(predicted_regr_values, y_test_split_initial_non_zero, c='k', label='data')       
#        pl.show()         
#        
                  
    average_mae = average_mae / 5.
    average_pearson = average_pearson / 5.
    print "Mean MAE: " + str(average_mae)   
    print "Mean Pearson: " + str(average_pearson) 


if __name__ == "__main__":

#    test_regression_models()

    test_regression_models_all_features()
    
    
