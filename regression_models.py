import pandas

import numpy as np

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import Imputer

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing as pre

import statsmodels.api as sm


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

train = pandas.read_csv('data/train_v2.csv')
test = pandas.read_csv('data/test_v2.csv')
train_loss = train.loss
#print train_loss

train = train[['f527','f528', 'f274']]
test = test[['f527','f528', 'f274']]

imp = Imputer()
imp.fit(train)

train = imp.transform(train)
test = imp.transform(test)

train=pre.StandardScaler().fit_transform(train)
test=pre.StandardScaler().fit_transform(test)

X_train,X_test,y_train,y_test = train_test_split( train, train_loss.apply(lambda x: 1 if x>0 else 0), test_size=0.3, random_state=42)

#clf = LogisticRegression(penalty='l1')
#
#clf.fit(X_train,y_train)
#
#print roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
#
#clf = LogisticRegression(penalty='l2')
#
#clf.fit(X_train,y_train)
#
#print roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])

clf = LogisticRegression(C=1e20,penalty='l2')

clf.fit(X_train,y_train)

print roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])

#glm = sm.GLM(y_train,X_train,sm.families.Binomial())
#
#results = glm.fit()
#
#print roc_auc_score(y_test,results.predict(X_test))


X_train,X_test,y_train,y_test = train_test_split( train, train_loss.apply(lambda x: 1 if x>0 else 0), test_size=0.0, random_state=42)
clf = LogisticRegression(C=1e20,penalty='l2')


clf.fit(X_train,y_train)
predsP = clf.predict(test)
print predsP

nztrain = np.where(y_train > 0)[0]

nztest = np.where(predsP == 1.)[0]

nztrain0 = np.where(y_train == 0.)[0]
nztest0 = np.where(predsP == 0.)[0]

#xtrainP = X_train[nztrain]

#Use all the features for the second classification:

train = pandas.read_csv('data/train_v2.csv')
test = pandas.read_csv('data/test_v2.csv')

print train.shape
cols = set(train.columns)
cols.remove('loss')
cols = list(cols)
train = train[cols]
print train.shape

imp = Imputer()
imp.fit(train)
train = imp.transform(train)
test = imp.transform(test)

train=pre.StandardScaler().fit_transform(train)
test=pre.StandardScaler().fit_transform(test)

X_train,X_test,y_train,y_test = train_test_split( train, train_loss.apply(lambda x: 1 if x>0 else 0), test_size=0.0, random_state=42)

xtrainP = X_train[nztrain]
xtestP = test[nztest]

ytrain0 = y_train[nztrain0]
ytrain1 = train_loss[nztrain]

print "Training second classifier..."

clf = LogisticRegression(C=1e10,penalty='l2')

clf.fit(xtrainP,ytrain1)

preds = clf.predict(xtestP)
  
print preds[:9]

print test.shape[0]

predsorig = np.asarray([0.] * test.shape[0])
predsorig[nztest] = preds
predsorig[nztest0] = 0.

np.savetxt('predictions/predictions.csv',predsorig ,delimiter = ',', fmt = '%d')

prediction_values_file = "predictions/predictions.csv"
sample_file = "data/sampleSubmission.csv"
competition_format_file = "predictions/predictions_competition_LogisticRegression_3_features_initial_all_after.csv"
write_prediction_file(prediction_values_file, sample_file, competition_format_file)

#print roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
