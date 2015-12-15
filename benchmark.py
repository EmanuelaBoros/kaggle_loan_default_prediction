"""

Beating the Benchmark :::::: Kaggle Loan Default Prediction Challenge.
__author__ : Abhishek

"""

import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn import preprocessing
from sklearn.svm import LinearSVC
import  scipy.stats as stats
import sklearn.linear_model as lm
import sklearn.svm as svm
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingClassifier

def testdata(filename):
	X = pd.read_table(filename, sep=',')

	X = np.asarray(X.values, dtype = float)

	col_mean = stats.nanmean(X,axis=0)
	inds = np.where(np.isnan(X))
	X[inds]=np.take(col_mean,inds[1])
	data = np.asarray(X[:,1:-3], dtype = float)

	return data
	
def data(filename):
	X = pd.read_table(filename, sep=',')

	X = np.asarray(X.values, dtype = float)

	col_mean = stats.nanmean(X,axis=0)
	inds = np.where(np.isnan(X))
	X[inds]=np.take(col_mean,inds[1])

	labels = np.asarray(X[:,-1], dtype = float)
	data = np.asarray(X[:,1:-4], dtype = float)
	return data, labels


def createSub(clf_binary, clf_regressor, traindata, labels, testdata, use_existing_svc_model):
    
	sub = 1

	labels = np.asarray(map(int,labels))

	niter = 10
	auc_list = []
	mean_auc = 0.0; itr = 0
	if sub == 1: 
 
		xtrain = traindata#[train]
		xtest = testdata#[test]

		ytrain = labels#[train]
		predsorig = np.asarray([0.] * testdata.shape[0]) #np.copy(ytest)

		labelsP = []

		for i in range(len(labels)):
			if labels[i] > 0:
				labelsP.append(1.)
			else:
				labelsP.append(0.)

		labelsP = np.asarray(labelsP)
		ytrainP = labelsP
  
		if use_existing_svc_model:
			lsvc = pickle.load(open("models/lsvc_c=0.01_penalty=l1,dual=false_run_1.pkl", "rb" ) )
			print "Loaded SVC model"
#			xtrainP = pickle.load(open("models/xtrainP_run_1.pkl", "rb" ) )
#   			xtestP = pickle.load(open("models/xtestP.pkl", "rb" ) )
		else:               

			lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, verbose = 2)  
  
			lsvc.fit(xtrain, ytrainP)  

			pickle.dump(lsvc, open('models/lsvc_c=0.01_penalty=l1,dual=false_run_2.pkl', 'wb'))
			print "Saved SVC model"
  
			xtrainP = lsvc.transform(xtrain)
			xtestP =  lsvc.transform(xtest)
#			pickle.dump(xtrainP, open('models/xtrainP_run_1.pkl', 'wb'))
#			pickle.dump(xtestP, open('models/xtestP.pkl', 'wb'))
  
		xtrainP = lsvc.transform(xtrain)
		xtestP =  lsvc.transform(xtest)
  
		xtrainP = xtrain
		xtestP = xtest
  
		clf_binary.fit(xtrainP,ytrainP)
		pickle.dump(clf_binary, open('models/clf_binary_SCV_C=1.0_kernel=poly_degree=3.pkl', 'wb'))
		print "Saved LogisticRegression model"
  
#		clf_binary = pickle.load(open('models/clf_binary_SCV_C=1.0_kernel=poly_degree=3.pkl', 'rb'))
		predsP = clf_binary.predict(xtestP)

		nztrain = np.where(ytrainP > 0)[0]
		nztest = np.where(predsP == 1.)[0]

		nztrain0 = np.where(ytrainP == 0.)[0]
		nztest0 = np.where(predsP == 0.)[0]

		xtrainP = xtrain[nztrain]
		xtestP = xtest[nztest]

		ytrain0 = ytrain[nztrain0]
		ytrain1 = ytrain[nztrain]

		clf_regressor.fit(xtrainP,ytrain1)
		pickle.dump(clf_regressor, open('clf_over_0_SCV_C=1.0_kernel=poly_degree=3.pkl', 'wb'))
  		print "Saved LinearRegression model"
#		
#		clf_regressor = pickle.load(open('models/LinearRegression_fit_intercept=True_normalize=False_run_1.pkl', 'rb'))
    
		preds = clf_regressor.predict(xtestP)
  
		print preds[:9]

		predsorig[nztest] = preds
		predsorig[nztest0] = 0.

		np.savetxt('predictions.csv',predsorig ,delimiter = ',', fmt = '%d')

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
        
    

if __name__ == '__main__':
	filename = 'train_v2.csv'
	X_test = testdata('test_v2.csv')

	X, labels = data(filename)
	
#	clf_binary = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
#                             C=1.0, fit_intercept=True, intercept_scaling=1.0, 
#                             class_weight=None, random_state=None)
                             
	clf_binary = svm.LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None)
    
#	clf_regressor = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
	clf_binary = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0, max_depth=1, random_state=0)
 
	clf_regressor =  clf_binary

	X = preprocessing.scale(X)	
	X_test = preprocessing.scale(X_test)

	use_existing_svc_model = False
	createSub(clf_binary, clf_regressor, X, labels, X_test, use_existing_svc_model)

	prediction_values_file = "predictions.csv"
	sample_file = "sampleSubmission.csv"
	competition_format_file = "predictions_competition_GradientBoostingClassifier_selected_features.csv"
	write_prediction_file(prediction_values_file, sample_file, competition_format_file)


