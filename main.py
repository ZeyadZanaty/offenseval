from DataReader import DataReader
from Preprocess import Preprocess

dr = DataReader('./datasets/training-v1/offenseval-training-v1.tsv')
data,labels = dr.get_labelled_data()
prp = Preprocess(data)
data = prp.clean(['remove_stopwords','lemmatize'])
print(data[0])