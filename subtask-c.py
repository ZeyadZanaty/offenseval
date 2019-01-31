from DataReader import DataReader
from Preprocessor import Preprocessor
from Vectorizer import Vectorizer
from Classifier import Classifier
from DeepLearning import DeepLearner
from sklearn.model_selection import train_test_split as split
import numpy as np

sub_c=['IND','GRP','OTH']

dr_tr = DataReader('./datasets/training-v1/offenseval-training-v1.tsv','C')
tr_data,tr_labels = dr_tr.get_labelled_data()
tr_data,tr_labels = dr_tr.upsample(tr_data,tr_labels,label=1)
tr_data,tr_labels = dr_tr.shuffle(tr_data,tr_labels,'random')
tr_data,tr_labels = dr_tr.upsample(tr_data,tr_labels,label=2)
tr_data,tr_labels = dr_tr.shuffle(tr_data,tr_labels,'random')
tr_data,tr_labels = dr_tr.upsample(tr_data,tr_labels,label=2)
tr_data,tr_labels = dr_tr.shuffle(tr_data,tr_labels,'random')

dr_tst = DataReader('./datasets/test-C/testset-taskc.tsv')
tst_data,tst_ids = dr_tst.get_test_data()

tr_data = tr_data[:]
tr_labels = tr_labels[:]

##### Naive Bayes - Stem - Count
prp = Preprocessor('remove_stopwords','stem')
tr_data_clean = prp.clean(tr_data)
tst_data_clean = prp.clean(tst_data)

vct = Vectorizer('count')
tr_vectors = vct.vectorize(tr_data_clean)
tst_vectors = vct.vectorize(tst_data_clean)

clf = Classifier('M-NaiveBayes')
tuned_accs = clf.tune(tr_vectors,tr_labels,{'alpha':[1,5,10],'fit_prior':[True,False]},best_only=False,scoring=None)
print('NB Tuned:',tuned_accs)

predictions = clf.predict(tst_vectors)
with open('subtask-C-test-NB.csv','w') as f:
    for i,id in enumerate(tst_ids):
        f.write(str(id)+','+str(sub_c[predictions[i]])+'\n')

##### Logistic Regression - Remove Stopwords - Count
prp = Preprocessor('remove_stopwords')
tr_data_clean = prp.clean(tr_data)
tst_data_clean = prp.clean(tst_data)

vct = Vectorizer('count')
tr_vectors = vct.vectorize(tr_data_clean)
tst_vectors = vct.vectorize(tst_data_clean)

clf = Classifier('LogisticRegression')
tuned_accs = clf.tune(tr_vectors,tr_labels,{'penalty':['l2'],'solver':['sag','newton-cg','lbfgs']},best_only=False,scoring=None)
print('LR Tuned:',tuned_accs)

predictions = clf.predict(tst_vectors)
with open('subtask-C-test-LR.csv','w') as f:
    for i,id in enumerate(tst_ids):
        f.write(str(id)+','+str(sub_c[predictions[i]])+'\n')

##### Random Forest - Lemmatize - Count
prp = Preprocessor('lemmatize')
tr_data_clean = prp.clean(tr_data)
tst_data_clean = prp.clean(tst_data)

vct = Vectorizer('count')
tr_vectors = vct.vectorize(tr_data_clean)
tst_vectors = vct.vectorize(tst_data_clean)

clf = Classifier('RandomForest')
tuned_accs = clf.tune(tr_vectors,tr_labels,{'n_estimators':[30,40,90,160]},best_only=False,scoring=None)
print('RF Tuned:',tuned_accs)

predictions = clf.predict(tst_vectors)
with open('subtask-C-test-RF.csv','w') as f:
    for i,id in enumerate(tst_ids):
        f.write(str(id)+','+str(sub_c[predictions[i]])+'\n')

#### Voting
import csv
from collections import Counter
n_samples = tst_vectors.shape[0]
files = ['subtask-C-test-RF.csv','subtask-C-test-LR.csv','subtask-C-test-NB.csv']
predictions = [[] for _ in range(n_samples)]
most_common_predictions = [None for _ in range(n_samples)]
for file_name in files:
    with open(file_name,encoding='utf8') as f:
        reader = csv.reader(f, delimiter=',')
        for j,line in enumerate(reader):
            predictions[j].append(line[1])

for i,prediction in enumerate(predictions):
    cnt = Counter(prediction)
    most_common_predictions[i] = cnt.most_common(1)[0][0]

with open('subtask-C-test-voting.csv','w') as f:
    for i,id in enumerate(tst_ids):
        f.write(str(id)+','+str(most_common_predictions[i])+'\n')