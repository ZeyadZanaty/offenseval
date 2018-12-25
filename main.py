from DataReader import DataReader
from Preprocessor import Preprocessor
from Vectorizer import Vectorizer
from Classifier import Classifier

dr = DataReader('./datasets/training-v1/offenseval-training-v1.tsv')
data,labels = dr.get_labelled_data()
data = data[:]
labels = labels[:]
prp = Preprocessor('remove_stopwords','lemmatize')
data = prp.clean(data)
# prp.word_cloud(labels,'OFF')
vct = Vectorizer('word2vec',train=True)
vectors = vct.vectorize(data)
tr_vectors,tr_labels = vectors[:len(vectors)//2], labels[:len(vectors)//2]
tst_vectors,tst_labels = vectors[len(vectors)//2:], labels[len(vectors)//2:]
clf = Classifier('KNN')
tuned_clf = clf.tune(tr_vectors,tr_labels,{'n_neighbors':[5]},best_only=True)
print(clf.score(tst_vectors,tst_labels))