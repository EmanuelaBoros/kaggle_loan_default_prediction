import pandas

import numpy as np

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import Imputer

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing as pre
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

import statsmodels.api as sm

import numpy as np
import pylab as pl
from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_absolute_error

   


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

train = pandas.read_csv('data/train_v2_sample_10k.csv')
test = pandas.read_csv('data/test_v2_sample_10k.csv')
train_loss = train.loss
#print train_loss

train = train[['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431', 'f757']]
test = test[['f527', 'f528', 'f274', 'f271', 'f2', 'f727', 'f337', 'f431', 'f757']]

#print train.shape
#cols = set(train.columns)
#cols.remove('loss')
#cols = list(cols)
#train = train[cols]
#print train.shape


imp = Imputer()
imp.fit(train)

train = imp.transform(train)
test = imp.transform(test)

train=pre.StandardScaler().fit_transform(train)
test=pre.StandardScaler().fit_transform(test)

####################################################"

#train  = np.asarray(train[:,1:-4], dtype = float)
#test  = np.asarray(test[:,1:-4], dtype = float)
#
#
#
#X_train,X_test,y_train,y_test = train_test_split( train, train_loss.apply(lambda x: 1 if x>0 else 0), test_size=0., random_state=42)

#
#lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, verbose = 1)
## 
#lsvc.fit(X_train, y_train)  
#train = lsvc.transform(train)
#test =  lsvc.transform(test)

#
#clf = ExtraTreesClassifier(n_estimators=1000, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', 
#                           bootstrap=False, oob_score=False, n_jobs=8, random_state=None, verbose=1, min_density=None, compute_importances=None)
#clf.fit(X_train, y_train)
#
#
#thresh = sorted(clf.feature_importances_, reverse = True)[10]
#
#train = clf.transform(train, threshold=thresh)
#test =  clf.transform(test, threshold=thresh)
#
#print train.shape 
#
#
#
#print(len(train))

####################################################"

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

probas_ = clf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

print tpr

selected_indexes_over = np.where(tpr > 0.85)[0]
selected_indexes_under = np.where(tpr < 0.86)[0]
selected_indexes = set(selected_indexes_over).intersection( set(selected_indexes_under) )
print len(selected_indexes)
selected_indexes = list(selected_indexes)
selected_threshold_index = selected_indexes[int(len(selected_indexes) / 2)]
print selected_threshold_index
selected_threshold = thresholds[selected_threshold_index]
print selected_threshold

pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic example')
pl.legend(loc="lower right")
pl.show()

#glm = sm.GLM(y_train,X_train,sm.families.Binomial())
#
#results = glm.fit()
#
#print roc_auc_score(y_test,results.predict(X_test))


X_train,X_test,y_train,y_test = train_test_split( train, train_loss.apply(lambda x: 1 if x>0 else 0), test_size=0.0, random_state=42)
clf = LogisticRegression(C=1e20,penalty='l2')


clf.fit(X_train,y_train)
predsP = clf.predict_proba(test)
print predsP

nztrain = np.where(y_train > 0)[0]

nztest = np.where(predsP > selected_threshold)[0]
print "Total with ones : " + str(len(nztest))
print nztest[:10]

nztrain0 = np.where(y_train == 0.)[0]
nztest0 = np.where(predsP <= selected_threshold)[0]
print len(nztest0)
print "Total with zeros : " + str(len(nztest0))

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

selected_threshold = 0.16446347363
predictionIndexes0 = np.where(predsP <= selected_threshold)[0]
predictionIndexes1 = np.where(predsP > selected_threshold)[0]

predsorig = np.asarray([0.] * test.shape[0])
predsorig[nztest] = 1.
predsorig[nztest0] = 0.
print predsorig[:20]

np.savetxt('predictions/predictions.csv',predsorig ,delimiter = ',', fmt = '%d')

prediction_values_file = "predictions/predictions.csv"
sample_file = "data/sampleSubmission.csv"
competition_format_file = "predictions/predictions_competition_LogisticRegression_9_features_0.85_threshold_predicted_ones.csv"
write_prediction_file(prediction_values_file, sample_file, competition_format_file)

#print roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
